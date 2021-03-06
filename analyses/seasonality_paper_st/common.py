# -*- coding: utf-8 -*-
import concurrent.futures
import importlib.util
import logging
import math
import os
import pickle
import re
import shelve
import sys
import warnings
from collections import defaultdict
from copy import copy, deepcopy
from datetime import datetime
from functools import partial, reduce, wraps
from itertools import combinations, islice, product
from operator import mul
from pathlib import Path
from pprint import pprint
from string import ascii_lowercase
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
from alepython.ale import _sci_format, ale_plot, first_order_ale_quant
from dask.distributed import Client
from dateutil.relativedelta import relativedelta
from hsluv import hsluv_to_rgb, rgb_to_hsluv
from iris.time import PartialDateTime
from joblib import Parallel, delayed, parallel_backend
from loguru import logger as loguru_logger
from matplotlib.colors import LogNorm, SymLogNorm, from_levels_and_colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pdpbox import pdp
from scipy.ndimage.morphology import binary_dilation
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
    update_nested_dict,
)

if "TQDMAUTO" in os.environ:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm

mpl.rc_file(Path(__file__).parent / "matplotlibrc")

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

orig_cube_plotting = cube_plotting


def cube_plotting(*args, **kwargs):
    """Modified cube plotting with default arguments."""
    kwargs = kwargs.copy()
    assert len(args) <= 1, "At most 1 positional argument supported."
    # Assume certain default kwargs, unless overriden.
    cbar_fmt = ticker.FuncFormatter(lambda x, pos: simple_sci_format(x))
    defaults = dict(
        coastline_kwargs=dict(linewidth=0.3),
        gridline_kwargs=dict(zorder=0, alpha=0.8, linestyle="--", linewidth=0.3),
        colorbar_kwargs=dict(format=cbar_fmt, pad=0.02),
    )
    kwargs = update_nested_dict(defaults, kwargs)
    return orig_cube_plotting(*args, **kwargs)


register_cl_backend()
PAPER_DIR = Path(__file__).resolve().parent
data_memory = get_memory(PAPER_DIR.name, backend="cloudpickle", verbose=2)

map_figure_saver_kwargs = {"dpi": 1200}


class PaperFigureSaver(FigureSaver):
    experiment = "common"

    @wraps(FigureSaver.__call__)
    def __call__(self, filenames=None, sub_directory=None, **kwargs):
        new_inst = super().__call__(
            filenames=filenames, sub_directory=sub_directory, **kwargs
        )
        new_inst.experiment = self.experiment
        return new_inst

    @wraps(FigureSaver.save_figure)
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
    directories=Path("~") / "tmp" / PAPER_DIR.name, debug=False
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
    "all": "ALL",
    "15_most_important": "TOP15",
    "no_temporal_shifts": "CURR",
    "fapar_only": "15VEG_FAPAR",
    "lai_only": "15VEG_LAI",
    "sif_only": "15VEG_SIF",
    "vod_only": "15VEG_VOD",
    "lagged_fapar_only": "CURRDD_FAPAR",
    "lagged_lai_only": "CURRDD_LAI",
    "lagged_sif_only": "CURRDD_SIF",
    "lagged_vod_only": "CURRDD_VOD",
    "best_top_15": "BEST15",
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

units = {
    "DD": "days",
    "SWI": r"$\mathrm{m}^3 \mathrm{m}^{-3}$",
    "MaxT": "K",
    "DTR": "K",
    "Lightning": r"$\mathrm{strokes}\ \mathrm{km}^{-2}$",
    "CROP": "1",
    "POPD": r"$\mathrm{inh}\ \mathrm{km}^{-2}$",
    "HERB": "1",
    "SHRUB": "1",
    "TREE": "1",
    "AGB": "r$\mathrm{kg}\ \mathrm{m}^{-2}$",
    "VOD": "1",
    "FAPAR": "1",
    "LAI": r"$\mathrm{m}^2\ \mathrm{m}^{-2}$",
    "SIF": "r$\mathrm{mW}\ \mathrm{m}^{-2}\ \mathrm{sr}^{-1}\ \mathrm{nm}^{-1}$",
}


def add_units(variables):
    """Add units to variables based on the `units` dict."""
    if isinstance(variables, str):
        return add_units([variables])[0]
    var_units = []
    for var in variables:
        matched_unit_vars = [
            unit_var for unit_var in units if re.search(unit_var, var) is not None
        ]
        assert (
            len(matched_unit_vars) == 1
        ), f"There should only be exactly 1 matching variable for '{var}'."
        var_units.append(f"{var} ({units[matched_unit_vars[0]]})")
    return var_units


# SHAP parameters.
shap_params = {"job_samples": 2000}  # Samples per job.
shap_interact_params = {
    "job_samples": 50,  # Samples per job.
    "max_index": 5999,  # Maximum job array index (inclusive).
}

# Feature importance shelve.
fi_shelve_file = str(Path(DATA_DIR) / PAPER_DIR.name / "feature_importances" / "frames")
Path(fi_shelve_file).parent.mkdir(parents=True, exist_ok=True)

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


def repl_fill_names_columns(df, inplace=False, sub=""):
    return df.rename(
        columns=dict(
            (orig, short)
            for orig, short in zip(df.columns, repl_fill_names(df.columns, sub=sub))
        ),
        inplace=inplace,
    )


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
        no_fill_feature_order[shorten_features(entry.strip(fill_name("")))] = counter
        counter += 1

# If BA is included, position it first.
no_fill_feature_order["GFED4 BA"] = -1
no_fill_feature_order["BA"] = -2

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
    n_jobs=1,
    include_first_order=False,
    figure_saver=None,
    plot_samples=True,
    figsize=None,
):
    model.n_jobs = n_jobs

    if figsize is None:
        figsize = (6.15, 3.17)

    cbar_width = 0.01

    x_coords = {}
    x_coords["ALE start"] = 0
    x_coords["ALE end"] = 0.42
    x_coords["ALE cbar start"] = x_coords["ALE end"] + 0.01
    x_coords["ALE cbar end"] = x_coords["ALE cbar start"] + cbar_width
    x_coords["Samples start"] = 0.65
    x_coords["Samples end"] = 0.9
    x_coords["Samples cbar start"] = x_coords["Samples end"] + 0.01
    x_coords["Samples cbar end"] = x_coords["Samples cbar start"] + cbar_width

    y_bottom = {
        "Samples": 1 / 3,  # Samples plot and cbar bottom.
    }
    cbar_height = {
        "ALE": 0.6,
        "Samples": 0.4,
    }

    top = 1

    fig = plt.figure(figsize=figsize)

    # ALE plot axes.
    ax = [
        fig.add_axes(
            [x_coords["ALE start"], 0, x_coords["ALE end"] - x_coords["ALE start"], top]
        )
    ]
    # ALE plot cbar axes.
    cax = [
        fig.add_axes(
            [
                x_coords["ALE cbar start"],
                top * (1 - cbar_height["ALE"]) / 2,
                x_coords["ALE cbar end"] - x_coords["ALE cbar start"],
                cbar_height["ALE"],
            ]
        )
    ]
    if plot_samples:
        # Samples plot axes.
        ax.append(
            fig.add_axes(
                [
                    x_coords["Samples start"],
                    y_bottom["Samples"],
                    x_coords["Samples end"] - x_coords["Samples start"],
                    top - y_bottom["Samples"],
                ]
            )
        )
        # Samples plot cbar axes.
        cax.append(
            fig.add_axes(
                [
                    x_coords["Samples cbar start"],
                    (y_bottom["Samples"] + top) / 2 - cbar_height["Samples"] / 2,
                    x_coords["Samples cbar end"] - x_coords["Samples cbar start"],
                    cbar_height["Samples"],
                ]
            )
        )

    with parallel_backend("threading", n_jobs=n_jobs):
        fig, axes, (quantiles_list, ale, samples) = ale_plot(
            model,
            train_set,
            features,
            bins=20,
            fig=fig,
            ax=ax[0],
            plot_quantiles=False,
            quantile_axis=True,
            plot_kwargs={
                "kind": "grid",
                "cmap": "inferno",
                "colorbar_kwargs": dict(
                    format=ticker.FuncFormatter(
                        lambda x, pos: simple_sci_format(x, precision=1)
                    ),
                    cax=cax[0],
                    label="ALE (BA)",
                ),
            },
            return_data=True,
            n_jobs=n_jobs,
            include_first_order=include_first_order,
        )

    for ax_key in ("ale", "quantiles_x"):
        if ax_key in axes:
            axes[ax_key].xaxis.set_tick_params(rotation=50)

    axes["ale"].set_aspect("equal")
    axes["ale"].set_xlabel(add_units(features[0]))
    axes["ale"].set_ylabel(add_units(features[1]))
    axes["ale"].set_title("")

    axes["ale"].xaxis.set_ticklabels(
        np.vectorize(partial(simple_sci_format, precision=1))(quantiles_list[0])
    )
    axes["ale"].yaxis.set_ticklabels(
        np.vectorize(partial(simple_sci_format, precision=1))(quantiles_list[1])
    )

    for tick in axes["ale"].xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment("right")

    if plot_samples:
        # Plotting samples.
        mod_quantiles_list = []
        for axis, quantiles in zip(("x", "y"), quantiles_list):
            inds = np.arange(len(quantiles))
            mod_quantiles_list.append(inds)
            ax[1].set(**{f"{axis}ticks": inds})
            ax[1].set(
                **{
                    f"{axis}ticklabels": np.vectorize(
                        partial(simple_sci_format, precision=1)
                    )(quantiles)
                }
            )
            for label in getattr(ax[1], f"{axis}axis").get_ticklabels()[1::2]:
                label.set_visible(False)

        samples_img = ax[1].pcolormesh(
            *mod_quantiles_list, samples.T, norm=SymLogNorm(linthresh=1)
        )

        @ticker.FuncFormatter
        def samples_colorbar_fmt(x, pos):
            if x < 0:
                raise ValueError("Samples cannot be -ve.")
            if np.isclose(x, 0):
                return "0"
            if np.log10(x).is_integer():
                return simple_sci_format(x)
            return ""

        fig.colorbar(
            samples_img,
            # XXX: old
            # ax=ax, shrink=0.6, pad=0.01, aspect=30,
            cax=cax[1],
            label="samples",
            format=samples_colorbar_fmt,
        )
        ax[1].xaxis.set_tick_params(rotation=50)
        for tick in ax[1].xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment("right")
        ax[1].set_aspect("equal")
        ax[1].set_xlabel(add_units(features[0]))
        ax[1].set_ylabel(add_units(features[1]))
        fig.set_constrained_layout_pads(
            w_pad=0.000, h_pad=0.000, hspace=0.0, wspace=0.015
        )

    if figure_saver is not None:
        if plot_samples:
            figure_saver.save_figure(
                fig,
                "__".join(features),
                sub_directory="2d_ale_first_order" if include_first_order else "2d_ale",
            )
        else:
            figure_saver.save_figure(
                fig,
                "__".join(features) + "_no_count",
                sub_directory="2d_ale_first_order_no_count"
                if include_first_order
                else "2d_ale_no_count",
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
    fig_name=None,
    fig=None,
    ax=None,
    xlabel=None,
    ylabel=None,
    title=None,
    n_jobs=1,
    verbose=False,
    figure_saver=None,
    CACHE_DIR=None,
    bins=20,
    x_rotation=20,
):
    if fig is None and ax is None:
        fig, ax = plt.subplots(
            figsize=(7, 3)
        )  # Make sure plot is plotted onto a new figure.
    elif fig is None:
        fig = ax.get_figure()
    if ax is None:
        ax = plt.axes()

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

    mod_quantiles = np.arange(len(quantiles))

    markers = ["o", "v", "^", "<", ">", "x", "+"]
    for feature, quantiles, ale, marker in zip(
        columns, quantile_list, ale_list, markers
    ):
        # Interpolate each of the quantiles relative to the accumulated final quantiles.
        ax.plot(
            np.interp(quantiles, final_quantiles, mod_quantiles),
            ale,
            marker=marker,
            label=feature,
        )

    ax.legend(loc="best", ncol=2)

    ax.set_xticks(mod_quantiles[::2])
    ax.set_xticklabels(_sci_format(final_quantiles[::2], scilim=0.6))
    ax.xaxis.set_tick_params(rotation=x_rotation)

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: simple_sci_format(x))
    )

    fig.suptitle(title)
    ax.set_xlabel(xlabel, va="center_baseline")
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
        match = re.search(target_feature + r"(\d+)M", feature)

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
    X_train,
    master_mask,
    shap_values,
    kind="train",
    additional_mask=None,
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
        )
        if kind == "mean":
            kwargs.update(
                cmap="Spectral_r",
                cmap_midpoint=0,
                cmap_symmetric=True,
            )
        if kind == "rel_std":
            kwargs.update(
                vmin=1e-2,
                vmax=10,
                extend="both",
                nbins=5,
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
        mm_valid_indices,
        **train_test_split_kwargs,
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
        ignore (iterable of str): Subsets of the above to ignore.

    Returns:
        dict of dict: Keys are the given `folders` and the loaded data types.

    """
    data = defaultdict(dict)
    if which == "all":
        which = ("offset_data", "model", "data_split", "model_scores")

    for experiment in folders:
        # Load the experiment module.
        spec = importlib.util.spec_from_file_location(
            f"{experiment}_specific",
            str(PAPER_DIR / experiment / "specific.py"),
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
                            "endog_data",
                            "exog_data",
                            "master_mask",
                            "filled_datasets",
                            "masked_datasets",
                            "land_mask",
                        ),
                        module.get_offset_data(),
                    )
                    if key not in ignore
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


def ba_plotting(predicted_ba, masked_val_data, figure_saver, cbar_label_x_offset=None):
    # date_str = "2010-01 to 2015-04"
    text_xy = (0.02, 0.935)

    fig, axes = plt.subplots(
        3,
        1,
        subplot_kw={"projection": ccrs.Robinson()},
        figsize=(5.1, (2.3 + 0.01) * 3),
        gridspec_kw={"hspace": 0.01, "wspace": 0.01},
    )

    # Plotting params.

    def get_plot_kwargs(cbar_label="Burned Area Fraction", **kwargs):
        defaults = dict(
            colorbar_kwargs={
                "label": cbar_label,
            },
            cmap="brewer_RdYlBu_11",
        )
        return update_nested_dict(defaults, kwargs)

    assert np.all(predicted_ba.mask == masked_val_data.mask)

    boundaries = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    cmap = "inferno"
    extend = "both"

    # Plotting observed.
    fig, cb0 = cube_plotting(
        masked_val_data,
        ax=axes[0],
        **get_plot_kwargs(
            cmap=cmap,
            #         title=f"Observed BA\n{date_str}",
            title="",
            boundaries=boundaries,
            extend=extend,
            cbar_label="Ob. BA",
        ),
        return_cbar=True,
    )

    # Plotting predicted.
    fig, cb1 = cube_plotting(
        predicted_ba,
        ax=axes[1],
        **get_plot_kwargs(
            cmap=cmap,
            #         title=f"Predicted BA\n{date_str}",
            title="",
            boundaries=boundaries,
            extend=extend,
            cbar_label="Pr. BA",
        ),
        return_cbar=True,
    )

    # frac_diffs = (masked_val_data - predicted_ba) / masked_val_data
    frac_diffs = np.mean(masked_val_data - predicted_ba, axis=0) / np.mean(
        masked_val_data, axis=0
    )

    # Plotting differences.
    diff_boundaries = [-1e1, -1e0, 0, 1e-1]
    extend = "both"

    fig, cb2 = cube_plotting(
        frac_diffs,
        ax=axes[2],
        **get_plot_kwargs(
            #         title=f"BA Discrepancy <(Obs. - Pred.) / Obs.> \n{date_str}",
            title="",
            cmap_midpoint=0,
            boundaries=diff_boundaries,
            cbar_label="<Ob. - Pr.)> / <Ob.>",
            extend=extend,
            colorbar_kwargs=dict(aspect=24, shrink=0.6, extendfrac=0.07),
            cmap="BrBG",
        ),
        return_cbar=True,
    )

    if cbar_label_x_offset is not None:
        # Manual control.
        max_x = 0
        for cb in (cb0, cb1, cb2):
            bbox = cb.ax.get_position()
            if bbox.xmax > max_x:
                max_x = bbox.xmax

        for cb in (cb0, cb1, cb2):
            bbox = cb.ax.get_position()
            mean_y = (bbox.ymin + bbox.ymax) / 2.0
            cb.ax.yaxis.set_label_coords(
                max_x + cbar_label_x_offset, mean_y, transform=fig.transFigure
            )
    else:
        fig.align_labels()

    for ax, title in zip(axes, ascii_lowercase):
        ax.text(*text_xy, f"({title})", transform=ax.transAxes)

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


class SetupFourMapAxes:
    """Context manager than handles construction and formatting of map axes.

    A single shared colorbar axis is created.

    Examples:
        >>> with SetupFourMapAxes() as (fig, axes, cax):  # doctest: +SKIP
        >>>     # Carry out plotting here.  # doctest: +SKIP
        >>> # Keep using `fig`, etc... here to carry out saving, etc...  # doctest: +SKIP
        >>> # It is important this is done after __exit__ is called!  # doctest: +SKIP

    """

    def __init__(self, figsize=(9.86, 4.93), cbar="vertical"):
        """Define basic parameters used to set up the figure and axes."""
        self.fig = plt.figure(figsize=figsize)
        self.cbar = cbar

        if self.cbar == "vertical":
            # Axis factor.
            af = 3
            nrows = 2 * af

            gs = self.fig.add_gridspec(
                nrows=nrows,
                ncols=2 * af + 2,
                width_ratios=[1 / af, 1 / af] * af + [0.001] + [0.02],
            )
            self.axes = [
                self.fig.add_subplot(
                    gs[i * af : (i + 1) * af, j * af : (j + 1) * af],
                    projection=ccrs.Robinson(),
                )
                for j, i in product(range(2), repeat=2)
            ]

            diff = 2
            assert (
                diff % 2 == 0
            ), f"Want an even diff for symmetric bar placement (got diff {diff})."
            cax_l = 0 + diff // 2
            cax_u = nrows - diff // 2

            self.cax = self.fig.add_subplot(gs[cax_l:cax_u, -1])
        elif self.cbar == "horizontal":
            # Axis factor.
            af = 3
            ncols = 2 * af

            gs = self.fig.add_gridspec(
                nrows=2 * af + 1,
                ncols=ncols,
                height_ratios=[1 / af, 1 / af] * af + [0.05],
            )
            self.axes = [
                self.fig.add_subplot(
                    gs[i * af : (i + 1) * af, j * af : (j + 1) * af],
                    projection=ccrs.Robinson(),
                )
                for j, i in product(range(2), repeat=2)
            ]

            diff = 4
            assert (
                diff % 2 == 0
            ), f"Want an even diff for symmetric bar placement (got diff {diff})."
            cax_l = 0 + diff // 2
            cax_u = ncols - diff // 2

            self.cax = self.fig.add_subplot(gs[-1, cax_l:cax_u])
        else:
            raise ValueError(f"Unkown value for 'cbar' {cbar}.")

    def __enter__(self):
        """Return the figure, 4 main plotting axes, and colorbar axes."""
        return self.fig, self.axes, self.cax

    def __exit__(self, exc_type, value, traceback):
        """Adjust axis positions after plotting."""
        if self.cbar == "vertical":
            self.fig.subplots_adjust(wspace=0, hspace=0.5)

            # Move the left-column axes to the right to decrease the gap.
            for ax in self.axes[0:2]:
                box = ax.get_position()
                shift = 0.015
                box.x0 += shift
                box.x1 += shift
                ax.set_position(box)
        elif self.cbar == "horizontal":
            self.fig.subplots_adjust(wspace=-0.43, hspace=0.5)

            self.cax.xaxis.set_label_position("top")

            # Move the legend axes upwards.
            ax = self.cax
            box = ax.get_position()
            shift = 0.01
            box.y0 += shift
            box.y1 += shift
            ax.set_position(box)
