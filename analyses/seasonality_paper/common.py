# -*- coding: utf-8 -*-
import concurrent.futures
import logging
import math
import os
import pickle
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from functools import partial, reduce, wraps
from itertools import combinations
from operator import mul
from pathlib import Path
from time import time

import cartopy.crs as ccrs
import cloudpickle
import dask.distributed
import eli5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import shap
from dask.distributed import Client
from hsluv import hsluv_to_rgb, rgb_to_hsluv
from joblib import Parallel, delayed, parallel_backend
from loguru import logger as loguru_logger
from matplotlib.colors import SymLogNorm, from_levels_and_colors
from matplotlib.patches import Rectangle
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from alepython.ale import _sci_format, ale_plot, first_order_ale_quant
from pdpbox import pdp
from wildfires.analysis import (
    FigureSaver,
    MidpointNormalize,
    constrained_map_plot,
    corr_plot,
    cube_plotting,
    data_processing,
    map_model_output,
    print_dataset_times,
    vif,
)
from wildfires.dask_cx1 import (
    CachedResults,
    DaskRandomForestRegressor,
    common_worker_threads,
    dask_fit_loco,
    fit_dask_sub_est_grid_search_cv,
    fit_dask_sub_est_random_search_cv,
    get_client,
    get_parallel_backend,
)
from wildfires.data import (
    DATA_DIR,
    HYDE,
    VODCA,
    WWLLN,
    AvitabileThurnerAGB,
    Copernicus_SWI,
    Datasets,
    ERA5_DryDayPeriod,
    ERA5_Temperature,
    ESA_CCI_Landcover_PFT,
    GFEDv4,
    GlobFluo_SIF,
    MOD15A2H_LAI_fPAR,
    dataset_times,
    dummy_lat_lon_cube,
    get_memory,
)
from wildfires.joblib.cloudpickle_backend import register_backend as register_cl_backend
from wildfires.logging_config import enable_logging
from wildfires.qstat import get_ncpus
from wildfires.utils import (
    NoCachedDataError,
    SimpleCache,
    Time,
    get_masked_array,
    get_unmasked,
    shorten_columns,
    shorten_features,
)

if "TQDMAUTO" in os.environ:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

loguru_logger.enable("alepython")
loguru_logger.remove()
loguru_logger.add(sys.stderr, level="WARNING")

logger = logging.getLogger(__name__)
enable_logging("jupyter")

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS.*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds.*")

warnings.filterwarnings(
    "ignore", 'Setting feature_perturbation = "tree_path_dependent".*'
)

normal_coast_linewidth = 0.3
mpl.rc("figure", figsize=(14, 6))
mpl.rc("font", size=9.0)

register_cl_backend()
PAPER_DIR = Path(__file__).resolve().parent
data_memory = get_memory(PAPER_DIR.name, backend="cloudpickle", verbose=2)

map_figure_saver_kwargs = {"dpi": 1200}

# 9 colors used to differentiate varying the lags throughout.
lags = [0, 1, 3, 6, 9, 12, 18, 24]
lag_colors = sns.color_palette("Set1", desat=0.85)
lag_color_dict = {lag: color for lag, color in zip(lags, lag_colors)}

experiments = ["all", "15_most_important", "no_temporal_shifts"]
experiment_colors = sns.color_palette("Set2")
experiment_color_dict = {
    experiment: color for experiment, color in zip(experiments, experiment_colors)
}
experiment_name_dict = {
    experiment: name
    for experiment, name in zip(experiments, ["all features", "top 15", "no lags"])
}
experiment_color_dict.update(
    {
        experiment_name_dict[experiment]: experiment_color_dict[experiment]
        for experiment in experiments
    }
)

experiment_markers = ["<", "o", ">"]
experiment_marker_dict = {
    experiment: marker for experiment, marker in zip(experiments, experiment_markers)
}
experiment_marker_dict.update(
    {
        experiment_name_dict[experiment]: experiment_marker_dict[experiment]
        for experiment in experiments
    }
)

# Creating the Data Structures used for Fitting
@data_memory.cache
def get_data(
    shift_months=[1, 3, 6, 9, 12, 18, 24], selection_variables=None, masks=None
):
    target_variable = "GFED4 BA"

    # Variables required for the above.
    required_variables = [target_variable]

    # Dataset selection.

    selection_datasets = [
        AvitabileThurnerAGB(),
        Copernicus_SWI(),
        ERA5_Temperature(),
        ESA_CCI_Landcover_PFT(),
        GFEDv4(),
        HYDE(),
        WWLLN(),
    ]
    # These datasets will potentially be shifted.
    datasets_to_shift = [
        ERA5_DryDayPeriod(),
        MOD15A2H_LAI_fPAR(),
        VODCA(),
        GlobFluo_SIF(),
    ]
    selection_datasets += datasets_to_shift
    if shift_months is not None:
        for shift in shift_months:
            for shift_dataset in datasets_to_shift:
                selection_datasets.append(
                    shift_dataset.get_temporally_shifted_dataset(
                        months=-shift, deep=False
                    )
                )

    if selection_variables is None:
        selection_variables = [
            "AGB Tree",
            "Diurnal Temp Range",
            "Dry Day Period",
            "FAPAR",
            "LAI",
            "Max Temp",
            "SIF",
            "SWI(1)",
            "ShrubAll",
            "TreeAll",
            "VOD Ku-band",
            "lightning",
            "pftCrop",
            "pftHerb",
            "popd",
        ]
        if shift_months is not None:
            for shift in shift_months:
                selection_variables.extend(
                    [
                        f"LAI {-shift} Month",
                        f"FAPAR {-shift} Month",
                        f"Dry Day Period {-shift} Month",
                        f"VOD Ku-band {-shift} Month",
                        f"SIF {-shift} Month",
                    ]
                )

    selection_variables = list(set(selection_variables).union(required_variables))

    selection = Datasets(selection_datasets).select_variables(selection_variables)
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = data_processing(
        selection,
        which="climatology",
        transformations={},
        deletions=[],
        use_lat_mask=False,
        use_fire_mask=False,
        target_variable=target_variable,
        masks=masks,
    )
    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


@data_memory.cache
def get_offset_data(
    shift_months=[1, 3, 6, 9, 12, 18, 24], selection_variables=None, masks=None
):
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = get_data(
        shift_months=shift_months, selection_variables=selection_variables, masks=masks
    )

    to_delete = []

    for column in exog_data:
        match = re.search(r"-\d{1,2}", column)
        if match:
            span = match.span()
            # Change the string to reflect the shift.
            original_offset = int(column[slice(*span)])
            if original_offset > -12:
                # Only shift months that are 12 or more months before the current month.
                continue
            comp = -(-original_offset % 12)
            new_column = " ".join(
                (
                    column[: span[0] - 1],
                    f"{original_offset} - {comp}",
                    column[span[1] + 1 :],
                )
            )
            if comp == 0:
                comp_column = column[: span[0] - 1]
            else:
                comp_column = " ".join(
                    (column[: span[0] - 1], f"{comp}", column[span[1] + 1 :])
                )
            print(column, comp_column)
            exog_data[new_column] = exog_data[column] - exog_data[comp_column]
            to_delete.append(column)

    for column in to_delete:
        del exog_data[column]

    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


def get_shap_values(rf, X, data=None, interaction=False):
    """Calculate SHAP values for `X`.

    When `data` is None, `feature_perturbation='tree_path_dependent'` by default.

    """
    if data is None:
        feature_perturbation = "tree_path_dependent"
    else:
        feature_perturbation = "interventional"

    explainer = shap.TreeExplainer(
        rf, data=data, feature_perturbation=feature_perturbation
    )

    if interaction:
        return explainer.shap_interaction_values(X)
    return explainer.shap_values(X)


def save_ale_2d_and_get_importance(
    model,
    train_set,
    features,
    n_jobs=8,
    include_first_order=False,
    figure_saver=None,
    plot_samples=True,
    figsize=None,
):
    model.n_jobs = n_jobs

    if figsize is None:
        if plot_samples:
            figsize = (10, 4.5)
        else:
            figsize = (7.5, 4.5)

    fig, ax = plt.subplots(
        1,
        2 if plot_samples else 1,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.7, 1]} if plot_samples else None,
        constrained_layout=True if plot_samples else False,
    )  # Make sure plot is plotted onto a new figure.
    with parallel_backend("threading", n_jobs=n_jobs):
        fig, axes, (quantiles_list, ale, samples) = ale_plot(
            model,
            train_set,
            features,
            bins=20,
            fig=fig,
            ax=ax[0] if plot_samples else ax,
            plot_quantiles=True,
            quantile_axis=True,
            plot_kwargs={
                "colorbar_kwargs": dict(
                    format="%.0e",
                    pad=0.02 if plot_samples else 0.09,
                    aspect=32,
                    shrink=0.85,
                    ax=ax[0] if plot_samples else ax,
                )
            },
            return_data=True,
            n_jobs=n_jobs,
            include_first_order=include_first_order,
        )

    # plt.subplots_adjust(top=0.89)
    for ax_key in ("ale", "quantiles_x"):
        axes[ax_key].xaxis.set_tick_params(rotation=45)

    if plot_samples:
        # Plotting samples.
        ax[1].set_title("Samples")
        # ax[1].set_xlabel(f"Feature '{features[0]}'")
        # ax[1].set_ylabel(f"Feature '{features[1]}'")
        mod_quantiles_list = []
        for axis, quantiles in zip(("x", "y"), quantiles_list):
            inds = np.arange(len(quantiles))
            mod_quantiles_list.append(inds)
            ax[1].set(**{f"{axis}ticks": inds})
            ax[1].set(**{f"{axis}ticklabels": _sci_format(quantiles, scilim=0.6)})
        samples_img = ax[1].pcolormesh(
            *mod_quantiles_list, samples.T, norm=SymLogNorm(linthresh=1)
        )
        fig.colorbar(samples_img, ax=ax, shrink=0.6, pad=0.01)
        ax[1].xaxis.set_tick_params(rotation=90)
        ax[1].set_aspect("equal")
        fig.set_constrained_layout_pads(
            w_pad=0.000, h_pad=0.000, hspace=0.0, wspace=0.015
        )

    if figure_saver is not None:
        figure_saver.save_figure(
            fig,
            "__".join(features),
            sub_directory="2d_ale_first_order" if include_first_order else "2d_ale",
        )

    #     min_samples = (
    #         train_set.shape[0] / reduce(mul, map(lambda x: len(x) - 1, quantiles_list))
    #     ) / 10
    #     return np.ma.max(ale[samples_grid > min_samples]) - np.ma.min(
    #         ale[samples_grid > min_samples]
    #     )

    return np.ptp(ale)


def save_pdp_plot_2d(model, X_train, features, n_jobs, figure_saver=None):
    model.n_jobs = n_jobs
    with parallel_backend("threading", n_jobs=n_jobs):
        pdp_interact_out = pdp.pdp_interact(
            model=model,
            dataset=X_train,
            model_features=X_train.columns,
            features=features,
            num_grid_points=[20, 20],
        )

    fig, axes = pdp.pdp_interact_plot(
        pdp_interact_out, features, x_quantile=True, figsize=(7, 8)
    )
    axes["pdp_inter_ax"].xaxis.set_tick_params(rotation=45)
    if figure_saver is not None:
        figure_saver.save_figure(fig, "__".join(features), sub_directory="pdp_2d")


def save_ale_plot_1d_with_ptp(
    model,
    X_train,
    column,
    n_jobs=8,
    monte_carlo_rep=1000,
    monte_carlo_ratio=100,
    verbose=False,
    monte_carlo=True,
    center=False,
    figure_saver=None,
):
    model.n_jobs = n_jobs
    with parallel_backend("threading", n_jobs=n_jobs):
        fig, ax = plt.subplots(
            figsize=(7.5, 4.5)
        )  # Make sure plot is plotted onto a new figure.
        out = ale_plot(
            model,
            X_train,
            column,
            bins=20,
            monte_carlo=monte_carlo,
            monte_carlo_rep=monte_carlo_rep,
            monte_carlo_ratio=monte_carlo_ratio,
            plot_quantiles=True,
            quantile_axis=True,
            rugplot_lim=0,
            scilim=0.6,
            return_data=True,
            return_mc_data=True,
            verbose=verbose,
            center=center,
        )
    if monte_carlo:
        fig, axes, data, mc_data = out
    else:
        fig, axes, data = out

    for ax_key in ("ale", "quantiles_x"):
        axes[ax_key].xaxis.set_tick_params(rotation=45)

    sub_dir = "ale" if monte_carlo else "ale_non_mc"
    if figure_saver is not None:
        figure_saver.save_figure(fig, column, sub_directory=sub_dir)

    if monte_carlo:
        mc_ales = np.array([])
        for mc_q, mc_ale in mc_data:
            mc_ales = np.append(mc_ales, mc_ale)
        return np.ptp(data[1]), np.ptp(mc_ales)
    else:
        return np.ptp(data[1])


def save_pdp_plot_1d(model, X_train, column, n_jobs, CACHE_DIR, figure_saver=None):
    data_file = os.path.join(CACHE_DIR, "pdp_data", column)

    if not os.path.isfile(data_file):
        model.n_jobs = n_jobs
        with parallel_backend("threading", n_jobs=n_jobs):
            pdp_isolate_out = pdp.pdp_isolate(
                model=model,
                dataset=X_train,
                model_features=X_train.columns,
                feature=column,
                num_grid_points=20,
            )
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        with open(data_file, "wb") as f:
            pickle.dump((column, pdp_isolate_out), f, -1)
    else:
        with open(data_file, "rb") as f:
            column, pdp_isolate_out = pickle.load(f)

    # With ICEs.
    fig_ice, axes_ice = pdp.pdp_plot(
        pdp_isolate_out,
        column,
        plot_lines=True,
        center=True,
        frac_to_plot=1000,
        x_quantile=True,
        figsize=(7, 5),
    )
    axes_ice["pdp_ax"].xaxis.set_tick_params(rotation=45)
    if figure_saver is not None:
        figure_saver.save_figure(fig_ice, column, sub_directory="pdp")

    # Without ICEs.
    fig_no_ice, ax = plt.subplots(figsize=(7.5, 4.5))
    plt.plot(pdp_isolate_out.pdp - pdp_isolate_out.pdp[0], marker="o")
    plt.xticks(
        ticks=range(len(pdp_isolate_out.pdp)),
        labels=_sci_format(pdp_isolate_out.feature_grids, scilim=0.6),
        rotation=45,
    )
    plt.xlabel(f"{column}")
    plt.title(f"PDP of feature '{column}'\nBins: {len(pdp_isolate_out.pdp)}")
    plt.grid(alpha=0.4, linestyle="--")
    if figure_saver is not None:
        figure_saver.save_figure(fig_no_ice, column, sub_directory="pdp_no_ice")
    return (fig_ice, fig_no_ice), pdp_isolate_out, data_file


def multi_ale_plot_1d(
    model,
    X_train,
    columns,
    fig_name,
    xlabel=None,
    ylabel=None,
    title=None,
    n_jobs=8,
    verbose=False,
    figure_saver=None,
):
    fig, ax = plt.subplots(
        figsize=(7.5, 4.5)
    )  # Make sure plot is plotted onto a new figure.
    model.n_jobs = n_jobs
    with parallel_backend("threading", n_jobs=n_jobs):
        quantile_list = []
        ale_list = []
        for feature in tqdm(
            columns, desc="Calculating feature ALEs", disable=not verbose
        ):
            quantiles, ale = first_order_ale_quant(
                model.predict, X_train, feature, bins=20
            )
            quantile_list.append(quantiles)
            ale_list.append(ale)

    # Construct quantiles from the individual quantiles, minimising the amount of interpolation.
    combined_quantiles = np.vstack([quantiles[None] for quantiles in quantile_list])

    final_quantiles = np.mean(combined_quantiles, axis=0)
    # Account for extrema.
    final_quantiles[0] = np.min(combined_quantiles)
    final_quantiles[-1] = np.max(combined_quantiles)

    mod_quantiles = np.arange(len(quantiles))
    for feature, quantiles, ale in zip(columns, quantile_list, ale_list):
        # Interpolate each of the quantiles relative to the accumulated final quantiles.
        ax.plot(
            np.interp(quantiles, final_quantiles, mod_quantiles),
            ale,
            marker="o",
            ms=3,
            label=feature,
        )

    ax.legend(loc="best")
    ax.set_xticks(mod_quantiles)
    ax.set_xticklabels(_sci_format(final_quantiles, scilim=0.6))
    ax.xaxis.set_tick_params(rotation=45)
    ax.grid(alpha=0.4, linestyle="--")

    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if figure_saver is not None:
        figure_saver.save_figure(fig, fig_name, sub_directory="multi_ale")


def add_common_path_deco(f):
    """Add path to 'common' before executing a function remotely."""
    COMMON_DIR = Path(__file__).resolve().parent
    # Adding wraps(f) here causes issues with an unmodified path.
    def path_f(*args, **kwargs):
        if sys.path[0] != str(COMMON_DIR):
            sys.path.insert(0, str(COMMON_DIR))
        # Call the original function.
        return f(*args, **kwargs)

    return path_f


def get_lag(feature, target_feature=None):
    """Return the lag duration as an integer.

    Optionally a specific target feature can be required.

    Args:
        feature (str): Feature to extract month from.
        target_feature (str): If given, this feature is required for a successful
            match.

    Returns:
        int or None: For successful matches (see `target_feature`), an int
            representing the lag duration is returned. Otherwise, `None` is returned.

    """
    if target_feature is None:
        target_feature = ".*"
    else:
        target_feature = re.escape(target_feature)

    match = re.search(target_feature + r"\s-(\d+)\s", feature)
    if match is not None:
        return int(match.groups(default="0")[0])
    if match is None and re.match(target_feature, feature):
        return 0
    return None


def filter_by_month(features, target_feature, max_month):
    """Filter feature names using a single target feature and maximum month.

    Args:
        features (iterable of str): Feature names to select from.
        target_feature (str): String in `features` to match against.
        max_month (int): Maximum month.

    Returns:
        iterable of str: The filtered feature names, subset of `features`.

    """
    filtered = []
    for feature in features:
        lag = get_lag(feature, target_feature=target_feature)
        if lag is not None and lag <= max_month:
            filtered.append(feature)
    return filtered
