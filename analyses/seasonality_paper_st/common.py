# -*- coding: utf-8 -*-
import concurrent.futures
import importlib.util
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
from itertools import combinations, islice, product
from operator import mul
from pathlib import Path
from time import time

import cartopy.crs as ccrs
import cloudpickle
import dask.distributed
import eli5
import iris
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import shap
from dask.distributed import Client
from dateutil.relativedelta import relativedelta
from hsluv import hsluv_to_rgb, rgb_to_hsluv
from iris.time import PartialDateTime
from joblib import Parallel, delayed, parallel_backend
from loguru import logger as loguru_logger
from matplotlib.colors import SymLogNorm, from_levels_and_colors
from matplotlib.lines import Line2D
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
    dask_fit_combinations,
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
    regrid,
)
from wildfires.joblib.cloudpickle_backend import register_backend as register_cl_backend
from wildfires.logging_config import enable_logging
from wildfires.qstat import get_ncpus
from wildfires.utils import (
    NoCachedDataError,
    SimpleCache,
    Time,
    ensure_datetime,
    get_batches,
    get_centres,
    get_local_extrema,
    get_local_maxima,
    get_local_minima,
    get_masked_array,
    get_unmasked,
    match_shape,
    replace_cube_coord,
    shorten_columns,
    shorten_features,
    significant_peak,
    simple_sci_format,
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


class PaperFigureSaver(FigureSaver):
    def save_figure(self, fig, filename, directory=None, sub_directory=None, **kwargs):
        # Make sure that the name of the experiment is present.
        filename = Path(filename)
        if not re.match(self.experiment + "_", filename.name):
            filename = filename.with_name("_".join((self.experiment, filename.name)))

        super().save_figure(
            fig,
            str(filename),
            directory=directory,
            sub_directory=sub_directory,
            **kwargs,
        )


figure_saver = PaperFigureSaver(
    directories=Path("~") / "tmp" / PAPER_DIR.name, debug=True
)
map_figure_saver = figure_saver(**map_figure_saver_kwargs)

# 9 colors used to differentiate varying the lags throughout.
lags = [0, 1, 3, 6, 9, 12, 18, 24]
lag_colors = sns.color_palette("Set1", desat=0.85)
lag_color_dict = {lag: color for lag, color in zip(lags, lag_colors)}

experiments = ["all", "15_most_important", "no_temporal_shifts", "best_top_15"]
experiment_colors = sns.color_palette("Set2")
experiment_color_dict = {
    experiment: color for experiment, color in zip(experiments, experiment_colors)
}
experiment_name_dict = {
    "all": "all",
    "best_top_15": "best top 15",
    "15_most_important": "top 15",
    "no_temporal_shifts": "no lags",
    "fapar_only": "best top 15 (fAPAR)",
    "sif_only": "best top 15 (SIF)",
    "lai_only": "best top 15 (LAI)",
    "vod_only": "best top 15 (VOD)",
    "lagged_fapar_only": "lagged fAPAR only",
    "lagged_sif_only": "lagged SIF only",
    "lagged_lai_only": "lagged LAI only",
    "lagged_vod_only": "lagged VOD only",
}
experiment_color_dict.update(
    {
        experiment_name_dict[experiment]: experiment_color_dict[experiment]
        for experiment in experiments
    }
)

experiment_markers = ["<", "o", ">", "x"]
experiment_marker_dict = {
    experiment: marker for experiment, marker in zip(experiments, experiment_markers)
}
experiment_marker_dict.update(
    {
        experiment_name_dict[experiment]: experiment_marker_dict[experiment]
        for experiment in experiments
    }
)

# SHAP parameters.
shap_params = {"job_samples": 2000}  # Samples per job.
shap_interact_params = {
    "job_samples": 50,  # Samples per job.
    "max_index": 5999,  # Maximum job array index (inclusive).
}

# Specify common RF (training) params.
n_splits = 5

default_param_dict = {"random_state": 1, "bootstrap": True}

param_dict = {
    **default_param_dict,
    "ccp_alpha": 2e-9,
    "max_depth": 18,
    "max_features": "auto",
    "min_samples_leaf": 3,
    "min_samples_split": 2,
    "n_estimators": 500,
}


def common_get_model(cache_dir, X_train=None, y_train=None):
    cached = CachedResults(
        estimator_class=DaskRandomForestRegressor,
        n_splits=n_splits,
        cache_dir=cache_dir,
    )
    model = DaskRandomForestRegressor(**param_dict)
    model_key = tuple(sorted(model.get_params().items()))
    try:
        model = cached.get_estimator(model_key)
    except KeyError:
        with parallel_backend("dask"):
            model.fit(X_train, y_train)
        cached.store_estimator(model_key, model)
    model.n_jobs = get_ncpus()
    return model


def common_get_model_scores(rf, X_test, X_train, y_test, y_train):
    rf.n_jobs = get_ncpus()
    with parallel_backend("threading", n_jobs=get_ncpus()):
        y_pred = rf.predict(X_test)
        y_train_pred = rf.predict(X_train)
    return {
        "test_r2": r2_score(y_test, y_pred),
        "test_mse": mean_squared_error(y_test, y_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "train_mse": mean_squared_error(y_train, y_train_pred),
    }


# Training and validation test splitting.
train_test_split_kwargs = dict(random_state=1, shuffle=True, test_size=0.3)


# Data filling params.
st_persistent_perc = 50
st_k = 4

filled_variables = {"SWI(1)", "FAPAR", "LAI", "VOD Ku-band", "SIF"}
filled_variables.update(shorten_features(filled_variables))


def fill_name(name):
    return f"{name} {st_persistent_perc}P {st_k}k"


def get_filled_names(names):
    if isinstance(names, str):
        return get_filled_names((names,))[0]
    filled = []
    for name in names:
        if any(var in name for var in filled_variables):
            filled.append(fill_name(name))
        else:
            filled.append(name)
    return filled


def repl_fill_name(name, sub=""):
    fill_ins = fill_name("")
    return name.replace(fill_ins, sub)


def repl_fill_names(names, sub=""):
    if isinstance(names, str):
        return repl_fill_names((names,), sub=sub)[0]
    return [repl_fill_name(name, sub=sub) for name in names]


feature_categories = {
    "meteorology": get_filled_names(
        ["Dry Day Period", "SWI(1)", "Max Temp", "Diurnal Temp Range", "lightning"]
    ),
    "human": ["pftCrop", "popd"],
    "landcover": ["pftHerb", "ShrubAll", "TreeAll", "AGB Tree"],
    "vegetation": get_filled_names(["VOD Ku-band", "FAPAR", "LAI", "SIF"]),
}

feature_order = {}
no_fill_feature_order = {}
counter = 0
for category, entries in feature_categories.items():
    for entry in entries:
        feature_order[entry] = counter
        no_fill_feature_order[entry.strip(fill_name(""))] = counter
        counter += 1

# If BA is included, position it first.
no_fill_feature_order["GFED4 BA"] = -1

# Creating the Data Structures used for Fitting
@data_memory.cache
def get_data(
    shift_months=[1, 3, 6, 9, 12, 18, 24],
    selection_variables=None,
    masks=None,
    st_persistent_perc=st_persistent_perc,
    st_k=st_k,
):
    target_variable = "GFED4 BA"

    # Variables required for the above.
    required_variables = [target_variable]

    # Dataset selection.

    selection_datasets = [
        AvitabileThurnerAGB(),
        ERA5_Temperature(),
        ESA_CCI_Landcover_PFT(),
        GFEDv4(),
        HYDE(),
        WWLLN(),
    ]

    # Datasets subject to temporal interpolation (filling).
    temporal_interp_datasets = [
        Datasets(Copernicus_SWI()).select_variables(("SWI(1)",)).dataset
    ]

    # Datasets subject to temporal interpolation and shifting.
    shift_and_interp_datasets = [
        Datasets(MOD15A2H_LAI_fPAR()).select_variables(("FAPAR", "LAI")).dataset,
        Datasets(VODCA()).select_variables(("VOD Ku-band",)).dataset,
        Datasets(GlobFluo_SIF()).select_variables(("SIF",)).dataset,
    ]

    # Datasets subject to temporal shifting.
    datasets_to_shift = [
        Datasets(ERA5_DryDayPeriod()).select_variables(("Dry Day Period",)).dataset
    ]

    all_datasets = (
        selection_datasets
        + temporal_interp_datasets
        + shift_and_interp_datasets
        + datasets_to_shift
    )

    # Determine shared temporal extent of the data.
    min_time, max_time = dataset_times(all_datasets)[:2]
    shift_min_time = min_time - relativedelta(years=2)

    # Sanity check.
    assert min_time == datetime(2010, 1, 1)
    assert shift_min_time == datetime(2008, 1, 1)
    assert max_time == datetime(2015, 4, 1)

    for dataset in datasets_to_shift + shift_and_interp_datasets:
        # Apply longer time limit to the datasets to be shifted.
        dataset.limit_months(shift_min_time, max_time)

        for cube in dataset:
            assert cube.shape[0] == 88

    for dataset in selection_datasets + temporal_interp_datasets:
        # Apply time limit.
        dataset.limit_months(min_time, max_time)

        if dataset.frequency == "monthly":
            for cube in dataset:
                assert cube.shape[0] == 64

    for dataset in all_datasets:
        # Regrid each dataset to the common grid.
        dataset.regrid()

    # Calculate and apply the shared mask.
    total_masks = []

    for dataset in temporal_interp_datasets + shift_and_interp_datasets:
        for cube in dataset.cubes:
            # Ignore areas that are always masked, e.g. water.
            ignore_mask = np.all(cube.data.mask, axis=0)

            # Also ignore those areas with low data availability.
            ignore_mask |= np.sum(cube.data.mask, axis=0) > (
                7 * 6  # Up to 6 months for each of the 7 complete years.
                + 10  # Additional Jan, Feb, Mar, Apr, + 6 extra.
            )

            total_masks.append(ignore_mask)

    combined_mask = reduce(np.logical_or, total_masks)

    # Apply mask to all datasets.
    for dataset in all_datasets:
        dataset.apply_masks(combined_mask)

    # Carry out the minima and season-trend filling.
    for datasets in (temporal_interp_datasets, shift_and_interp_datasets):
        for i, dataset in enumerate(datasets):
            datasets[i] = dataset.get_persistent_season_trend_dataset(
                persistent_perc=st_persistent_perc, k=st_k
            )

    datasets_to_shift.extend(shift_and_interp_datasets)
    selection_datasets += datasets_to_shift
    selection_datasets += temporal_interp_datasets

    if shift_months is not None:
        for shift in shift_months:
            for shift_dataset in datasets_to_shift:
                selection_datasets.append(
                    shift_dataset.get_temporally_shifted_dataset(
                        months=-shift, deep=False
                    )
                )

    if selection_variables is None:
        selection_variables = get_filled_names(
            [
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
        )
        if shift_months is not None:
            for shift in shift_months:
                selection_variables.extend(
                    [
                        f"{var} {-shift} Month"
                        for var in get_filled_names(
                            ["LAI", "FAPAR", "Dry Day Period", "VOD Ku-band", "SIF"]
                        )
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
    shift_months=[1, 3, 6, 9, 12, 18, 24],
    selection_variables=None,
    masks=None,
    st_persistent_perc=st_persistent_perc,
    st_k=st_k,
):
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = get_data(
        shift_months=shift_months,
        selection_variables=selection_variables,
        masks=masks,
        st_persistent_perc=st_persistent_perc,
        st_k=st_k,
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
    CACHE_DIR=None,
    bins=20,
):
    fig, ax = plt.subplots(
        figsize=(7, 3)
    )  # Make sure plot is plotted onto a new figure.

    quantile_list = []
    ale_list = []
    for feature in tqdm(columns, desc="Calculating feature ALEs", disable=not verbose):
        cache = SimpleCache(
            f"{feature}_ale_{bins}",
            cache_dir=CACHE_DIR / "ale",
            verbose=10 if verbose else 0,
        )
        try:
            quantiles, ale = cache.load()
        except NoCachedDataError:
            model.n_jobs = n_jobs

            with parallel_backend("threading", n_jobs=n_jobs):
                quantiles, ale = first_order_ale_quant(
                    model.predict, X_train, feature, bins=bins
                )
                cache.save((quantiles, ale))

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

    ax.legend(loc="best", ncol=2)

    ax.set_xticks(mod_quantiles[::2])
    ax.set_xticklabels(_sci_format(final_quantiles[::2], scilim=0.6))
    ax.xaxis.set_tick_params(rotation=20)

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: simple_sci_format(x))
    )

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
        if str(COMMON_DIR) not in sys.path:
            sys.path.insert(0, str(COMMON_DIR))
        # Call the original function.
        return f(*args, **kwargs)

    return path_f


def add_common_path(client):
    def _add_common():
        if str(PAPER_DIR) not in sys.path:
            sys.path.insert(0, str(PAPER_DIR))

    client.run(_add_common)


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
        target_feature = ".*?"
    else:
        target_feature = re.escape(target_feature)

    # Avoid dealing with the fill naming.
    feature = repl_fill_name(feature)

    match = re.search(target_feature + r"\s-(\d+)\s", feature)

    if match is None:
        # Try matching to 'short names'.
        match = re.search(target_feature + r"(\d+)\sM", feature)

    if match is not None:
        return int(match.groups(default="0")[0])
    if match is None and re.match(target_feature, feature):
        return 0
    return None


def get_lags(features, target_feature=None):
    if not isinstance(features, str):
        return [get_lag(feature, target_feature=target_feature) for feature in features]
    return get_lag(feature, target_feature=target_feature)


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


def sort_experiments(experiments):
    """Sort experiments based on `experiment_name_dict`."""
    name_lists = (
        list(experiment_name_dict.keys()),
        list(experiment_name_dict.values()),
    )
    order = []
    experiments = list(experiments)
    for experiment in experiments:
        for name_list in name_lists:
            if experiment in name_list:
                order.append(name_list.index(experiment))
                break
        else:
            # No break encountered, so no order could be found.
            raise ValueError(f"Experiment {experiment} could not be found.")
    out = []
    for i in np.argsort(order):
        out.append(experiments[i])
    return out


def sort_features(features):
    """Sort feature names using their names and shift magnitudes.

    Args:
        features (iterable of str): Feature names to sort.

    Returns:
        list of str: Sorted list of features.

    """
    raw_features = []
    lags = []
    for feature in features:
        lag = get_lag(feature)
        assert lag is not None
        # Remove fill naming addition.
        feature = repl_fill_name(feature)
        if str(lag) in feature:
            # Strip lag information from the string.
            raw_features.append(feature[: feature.index(str(lag))].strip("-").strip())
        else:
            raw_features.append(feature)
        lags.append(lag)
    sort_tuples = tuple(zip(features, raw_features, lags))
    return [
        s[0]
        for s in sorted(
            sort_tuples, key=lambda x: (no_fill_feature_order[x[1]], abs(int(x[2])))
        )
    ]


def transform_series_sum_norm(x):
    x = x / np.sum(np.abs(x))
    return x


def plot_and_list_importances(importances, methods, print_n=15, N=15, verbose=True):
    fig, ax = plt.subplots()

    transformed = {}

    combined = None
    for method in methods:
        transformed[method] = transform_series_sum_norm(importances[method])
        if combined is None:
            combined = transformed[method].copy()
        else:
            combined += transformed[method]
    combined.sort_values(ascending=False, inplace=True)

    transformed = pd.DataFrame(transformed).reindex(combined.index, axis=0)

    for method, marker in zip(methods, ["o", "x", "s", "^"]):
        ax.plot(
            transformed[method], linestyle="", marker=marker, label=method, alpha=0.8
        )
    ax.set_xticklabels(
        transformed.index, rotation=45 if len(transformed.index) <= 15 else 90
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.4)
    if verbose:
        print(combined[:print_n].to_latex())
    return combined[:N]


def calculate_2d_masked_shap_values(
    X_train, master_mask, shap_values, kind="train", additional_mask=None,
):
    shap_results = {}

    def time_abs_max(x):
        out = np.take_along_axis(
            x, np.argmax(np.abs(x), axis=0).reshape(1, *x.shape[1:]), axis=0
        )
        assert out.shape[0] == 1
        return out[0]

    agg_keys, agg_funcs = zip(
        ("masked_shap_arrs", lambda arr: np.mean(arr, axis=0)),
        ("masked_shap_arrs_std", lambda arr: np.std(arr, axis=0)),
        ("masked_abs_shap_arrs", lambda arr: np.mean(np.abs(arr), axis=0)),
        ("masked_abs_shap_arrs_std", lambda arr: np.std(np.abs(arr), axis=0)),
        ("masked_max_shap_arrs", time_abs_max),
    )
    for key in agg_keys:
        shap_results[key] = dict(data=[], vmins=[], vmaxs=[])

    mm_valid_indices, mm_valid_train_indices, mm_valid_val_indices = get_mm_indices(
        master_mask
    )
    if kind == "train":
        mm_kind_indices = mm_valid_train_indices[: shap_values.shape[0]]
    elif kind == "val":
        mm_kind_indices = mm_valid_val_indices[: shap_values.shape[0]]
    else:
        raise ValueError(f"Unknown kind: {kind}.")

    for i in tqdm(range(len(X_train.columns)), desc="Aggregating SHAP values"):
        # Convert 1D shap values into 3D array (time, lat, lon).
        masked_shap_comp = np.ma.MaskedArray(
            np.zeros_like(master_mask, dtype=np.float64), mask=np.ones_like(master_mask)
        )
        masked_shap_comp.ravel()[mm_kind_indices] = shap_values[:, i]

        if additional_mask is not None:
            masked_shap_comp.mask |= match_shape(
                additional_mask, masked_shap_comp.shape
            )

        # Calculate different aggregations over time.

        for key, agg_func in zip(agg_keys, agg_funcs):
            agg_shap = agg_func(masked_shap_comp)
            shap_results[key]["data"].append(agg_shap)
            shap_results[key]["vmins"].append(np.min(agg_shap))
            shap_results[key]["vmaxs"].append(np.max(agg_shap))

    # Calculate relative standard deviations.

    rel_agg_keys = [
        "masked_shap_arrs_rel_std",
        "masked_abs_shap_arrs_rel_std",
    ]
    rel_agg_sources_keys = [
        ("masked_shap_arrs", "masked_shap_arrs_std"),
        ("masked_abs_shap_arrs", "masked_abs_shap_arrs_std"),
    ]
    for rel_agg_key, rel_agg_sources_key in zip(rel_agg_keys, rel_agg_sources_keys):
        shap_results[rel_agg_key] = dict(data=[], vmins=[], vmaxs=[])
        for i in range(len(X_train.columns)):
            rel_agg_shap = shap_results[rel_agg_sources_key[1]]["data"][i] / np.ma.abs(
                shap_results[rel_agg_sources_key[0]]["data"][i]
            )
            shap_results[rel_agg_key]["data"].append(rel_agg_shap)
            shap_results[rel_agg_key]["vmins"].append(np.min(rel_agg_shap))
            shap_results[rel_agg_key]["vmaxs"].append(np.max(rel_agg_shap))

    for key, values in shap_results.items():
        values["vmin"] = min(values["vmins"])
        values["vmax"] = max(values["vmaxs"])

    return shap_results


def plot_shap_value_maps(
    X_train, shap_results, map_figure_saver, directory="shap_maps", close=True
):
    """

    Args:
        X_train (pandas DataFrame):
        shap_results (SHAP results dict from `calculate_2d_masked_shap_values`):
        map_figure_saver (FigureSaver instance):
        directory (str or Path): Figure saving directory.
        close (bool): If True, close figures after saving.

    """
    # Define common plotting profiles, as `cube_plotting` kwargs.

    def get_plot_kwargs(feature, results_dict, title_stub, kind=None):
        kwargs = dict(
            fig=plt.figure(figsize=(5.1, 2.8)),
            title=f"{title_stub} '{shorten_features(feature)}'",
            nbins=7,
            vmin=results_dict["vmin"],
            vmax=results_dict["vmax"],
            log=True,
            log_auto_bins=False,
            extend="neither",
            min_edge=1e-3,
            cmap="inferno",
            colorbar_kwargs={
                "format": "%0.1e",
                "label": f"SHAP ('{shorten_features(feature)}')",
            },
            coastline_kwargs={"linewidth": 0.3},
        )
        if kind == "mean":
            kwargs.update(
                cmap="Spectral_r", cmap_midpoint=0, cmap_symmetric=True,
            )
        if kind == "rel_std":
            kwargs.update(
                vmin=1e-2, vmax=10, extend="both", nbins=5,
            )
        return kwargs

    for i, feature in enumerate(tqdm(X_train.columns, desc="Mapping SHAP values")):
        for agg_key, title_stub, kind, sub_directory in (
            ("masked_shap_arrs", "Mean SHAP value for", "mean", "mean"),
            ("masked_shap_arrs_std", "STD SHAP value for", None, "std"),
            (
                "masked_shap_arrs_rel_std",
                "Rel STD SHAP value for",
                "rel_std",
                "rel_std",
            ),
            ("masked_abs_shap_arrs", "Mean |SHAP| value for", None, "abs_mean"),
            ("masked_abs_shap_arrs_std", "STD |SHAP| value for", None, "abs_std"),
            (
                "masked_abs_shap_arrs_rel_std",
                "Rel STD |SHAP| value for",
                "rel_std",
                "rel_abs_std",
            ),
            ("masked_max_shap_arrs", "Max || SHAP value for", "mean", "max"),
        ):
            fig = cube_plotting(
                shap_results[agg_key]["data"][i],
                **get_plot_kwargs(
                    feature,
                    results_dict=shap_results[agg_key],
                    title_stub=title_stub,
                    kind=kind,
                ),
            )
            map_figure_saver.save_figure(
                fig,
                f"{agg_key}_{feature}",
                sub_directory=Path(directory) / sub_directory,
            )
            if close:
                plt.close(fig)


def get_mm_indices(master_mask):
    mm_valid_indices = np.where(~master_mask.ravel())[0]
    mm_valid_train_indices, mm_valid_val_indices = train_test_split(
        mm_valid_indices, **train_test_split_kwargs,
    )
    return mm_valid_indices, mm_valid_train_indices, mm_valid_val_indices


def get_mm_data(x, master_mask, kind):
    """Return masked master_mask copy and training or validation indices.

    The master_mask copy is filled using the given data.

    Args:
        x (array-like): Data to use.
        master_mask (array):
        kind ({'train', 'val'})

    Returns:
        masked_data, mm_indices:

    """
    mm_valid_indices, mm_valid_train_indices, mm_valid_val_indices = get_mm_indices(
        master_mask
    )
    masked_data = np.ma.MaskedArray(
        np.zeros_like(master_mask, dtype=np.float64), mask=np.ones_like(master_mask)
    )
    if kind == "train":
        masked_data.ravel()[mm_valid_train_indices] = x
    elif kind == "val":
        masked_data.ravel()[mm_valid_val_indices] = x
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return masked_data


def load_experiment_data(folders, which="all", ignore=()):
    """Load data from specified experiments.

    Args:
        folders (iterable of {str, Path}): Folder names corresponding to the
            experiments to load data for.
        which (iterable of {'all', 'offset_data', 'model', 'data_split', 'model_scores'}):
            'all' loads everything.
        ignore (iterable of str): Subsets of the above the ignore.

    Returns:
        dict of dict: Keys are the given `folders` and the loaded data types.

    """
    data = defaultdict(dict)
    if which == "all":
        which = ("offset_data", "model", "data_split", "model_scores")

    for experiment in folders:
        # Load the experiment module.
        spec = importlib.util.spec_from_file_location(
            f"{experiment}_specific", str(PAPER_DIR / experiment / "specific.py"),
        )
        module = importlib.util.module_from_spec(spec)
        data[experiment]["module"] = module
        # Load module contents.
        spec.loader.exec_module(module)

        if "offset_data" in which:
            data[experiment].update(
                {
                    key: data
                    for key, data in zip(
                        (
                            key
                            for key in (
                                "endog_data",
                                "exog_data",
                                "master_mask",
                                "filled_datasets",
                                "masked_datasets",
                                "land_mask",
                            )
                            if key not in ignore
                        ),
                        module.get_offset_data(),
                    )
                }
            )
        if "model" in which:
            data[experiment]["model"] = module.get_model()
        if "data_split" in which:
            data[experiment].update(
                {
                    key: data
                    for key, data in zip(
                        (
                            key
                            for key in ("X_train", "X_test", "y_train", "y_test")
                            if key not in ignore
                        ),
                        module.data_split_cache.load(),
                    )
                }
            )
        if "model_scores" in which:
            data[experiment].update(module.get_model_scores())

    return data


def ba_plotting(predicted_ba, masked_val_data, figure_saver):
    # date_str = "2010-01 to 2015-04"
    text_xy = (0.012, 0.95)
    fs = 11

    fig, axes = plt.subplots(
        3,
        1,
        subplot_kw={"projection": ccrs.Robinson()},
        figsize=(5.1, 2.3 * 3),
        gridspec_kw={"hspace": 0.01, "wspace": 0.01},
    )

    # Plotting params.

    cbar_fmt = ticker.FuncFormatter(lambda x, pos: simple_sci_format(x))

    def get_plot_kwargs(cbar_label="Burned Area Fraction", **kwargs):
        colorbar_kwargs = kwargs.pop("colorbar_kwargs", {})
        return {
            **dict(
                colorbar_kwargs={
                    "label": cbar_label,
                    "format": cbar_fmt,
                    **colorbar_kwargs,
                },
                coastline_kwargs={"linewidth": 0.3},
                cmap="brewer_RdYlBu_11",
            ),
            **kwargs,
        }

    assert np.all(predicted_ba.mask == masked_val_data.mask)

    boundaries = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    cmap = "YlOrRd"
    extend = "both"

    # Plotting observed.
    fig = cube_plotting(
        masked_val_data,
        ax=axes[0],
        **get_plot_kwargs(
            cmap=cmap,
            #         title=f"Observed BA\n{date_str}",
            title="",
            boundaries=boundaries,
            extend=extend,
        ),
    )
    axes[0].text(*text_xy, "(a)", transform=axes[0].transAxes, fontsize=fs)

    # Plotting predicted.
    fig = cube_plotting(
        predicted_ba,
        ax=axes[1],
        **get_plot_kwargs(
            cmap=cmap,
            #         title=f"Predicted BA\n{date_str}",
            title="",
            boundaries=boundaries,
            extend=extend,
        ),
    )
    axes[1].text(*text_xy, "(b)", transform=axes[1].transAxes, fontsize=fs)

    frac_diffs = (masked_val_data - predicted_ba) / masked_val_data

    # Plotting differences.
    diff_boundaries = [-1e1, -1e0, 0, 1e-1]
    extend = "both"

    fig = cube_plotting(
        frac_diffs,
        ax=axes[2],
        **get_plot_kwargs(
            #         title=f"BA Discrepancy <(Obs. - Pred.) / Obs.> \n{date_str}",
            title="",
            cmap_midpoint=0,
            boundaries=diff_boundaries,
            colorbar_kwargs={"label": "(Obs. - Pred.) / Obs."},
            extend=extend,
        ),
    )
    axes[2].text(*text_xy, "(c)", transform=axes[2].transAxes, fontsize=fs)

    # Plot relative MSEs.
    """
    rel_mse = frac_diffs ** 2

    # Plotting differences.
    diff_boundaries = [1e-1, 1, 1e1, 1e2, 1e3]
    extend = "both"

    fig = cube_plotting(
        rel_mse,
        **get_plot_kwargs(
            cmap="inferno",
    #         title=r"BA Discrepancy <$\mathrm{((Obs. - Pred.) / Obs.)}^2$>" + f"\n{date_str}",
            title='',
            boundaries=diff_boundaries,
            colorbar_kwargs={"label": "1"},
            extend=extend,
        ),
    )
    plt.gca().text(*text_xy, '(d)', transform=plt.gca().transAxes, fontsize=fs)

    """
    figure_saver.save_figure(fig, f"ba_prediction", sub_directory="predictions")
