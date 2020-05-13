# -*- coding: utf-8 -*-
"""Investigate model fits with time lags and fire season masking.

"""
import logging
import math
import os
import warnings
from collections import namedtuple
from copy import deepcopy

import matplotlib as mpl
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import wildfires.analysis
from wildfires.analysis import *
from wildfires.data import *
from wildfires.logging_config import enable_logging
from wildfires.qstat import get_ncpus
from wildfires.utils import *

logger = logging.getLogger(__name__)
enable_logging()

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds*")

memory = get_memory("analysis_time_lags_fire_season", verbose=100)

FigureSaver.debug = True
FigureSaver.directory = os.path.expanduser(
    os.path.join("~", "tmp", "time_lags_fire_season")
)
os.makedirs(FigureSaver.directory, exist_ok=True)

normal_coast_linewidth = 0.5
mpl.rc("figure", figsize=(14, 6))
mpl.rc("font", size=9.0)

np.random.seed(1)

ba_dataset = "GFEDv4"

fit_data = namedtuple(
    "FitData",
    (
        "endog_data",
        "exog_data",
        "master_mask",
        "filled_datasets",
        "masked_datasets",
        "land_mask",
    ),
)


n_jobs = 5
with parallel_backend(
    "loky", n_jobs=n_jobs, inner_max_num_threads=math.floor(get_ncpus() / n_jobs)
):
    outputs = thres_fire_season_stats(0.1)

fire_season_mask = [out for out in outputs if out[0] == ba_dataset][0][4]

shift_months = [1, 3, 6, 12, 24]

selection_variables = (
    "VOD Ku-band -3 Month",
    "SIF",
    "VOD Ku-band -1 Month",
    "Dry Day Period -3 Month",
    "FAPAR",
    "pftHerb",
    "LAI -1 Month",
    "popd",
    "Dry Day Period -24 Month",
    "pftCrop",
    "FAPAR -1 Month",
    "FAPAR -24 Month",
    "Max Temp",
    "Dry Day Period -6 Month",
    "VOD Ku-band -6 Month",
)

ext_selection_variables = selection_variables + (
    "Dry Day Period -1 Month",
    "FAPAR -6 Month",
    "ShrubAll",
    "SWI(1)",
    "TreeAll",
)

in_fire_season = fit_data(
    *wildfires.analysis.time_lags.get_data(
        shift_months=[1, 3, 6, 12, 24],
        selection_variables=ext_selection_variables,
        masks=[~fire_season_mask],
    )
)

with FigureSaver("mean_ba_in_season"):
    cube_plotting(
        in_fire_season.filled_datasets[ba_dataset].cube,
        log=True,
        cmap="brewer_RdYlBu_11_r",
        coastline_kwargs=dict(linewidth=0.5),
        label="Burned Area (1)",
        title=f"{ba_dataset} In Season",
    )

out_fire_season = fit_data(
    *wildfires.analysis.time_lags.get_data(
        shift_months=[1, 3, 6, 12, 24],
        selection_variables=ext_selection_variables,
        masks=[fire_season_mask],
    )
)

with FigureSaver("mean_ba_out_season"):
    cube = deepcopy(out_fire_season.filled_datasets[ba_dataset].cube)
    target_shape = cube.shape
    cube.data.mask |= match_shape(
        np.all(np.isclose(cube.data.data, 0), axis=0), target_shape
    )
    cube_plotting(
        cube,
        log=True,
        cmap="brewer_RdYlBu_11_r",
        coastline_kwargs=dict(linewidth=0.5),
        label="Burned Area (1)",
        title=f"{ba_dataset} Out of Season",
    )

all_season = fit_data(
    *wildfires.analysis.time_lags.get_data(
        shift_months=[1, 3, 6, 12, 24], selection_variables=ext_selection_variables
    )
)

with FigureSaver("mean_ba_all_season"):
    cube = all_season.filled_datasets[ba_dataset].cube
    cube.data.mask |= match_shape(
        np.all(np.isclose(cube.data.data, 0), axis=0), target_shape
    )
    cube_plotting(
        cube,
        log=True,
        cmap="brewer_RdYlBu_11_r",
        coastline_kwargs=dict(linewidth=0.5),
        label="Burned Area (1)",
        title=f"{ba_dataset} All",
    )


splits = namedtuple("Splits", ["X_train", "X_test", "y_train", "y_test"])

in_fs_splits = splits(
    *train_test_split(
        in_fire_season.exog_data,
        in_fire_season.endog_data,
        random_state=1,
        shuffle=True,
        test_size=0.3,
    )
)

out_fs_splits = splits(
    *train_test_split(
        out_fire_season.exog_data,
        out_fire_season.endog_data,
        random_state=1,
        shuffle=True,
        test_size=0.3,
    )
)

all_splits = splits(
    *train_test_split(
        all_season.exog_data,
        all_season.endog_data,
        random_state=1,
        shuffle=True,
        test_size=0.3,
    )
)

rf_params = dict(
    max_depth=None,
    min_samples_leaf=3,
    min_samples_split=2,
    n_estimators=100,
    random_state=1,
)


@memory.cache
def fit_rf_in_season(**rf_params):
    regr = RandomForestRegressor(**rf_params)
    # Make sure all cores are used.
    regr.n_jobs = get_ncpus()
    regr.fit(in_fs_splits.X_train, in_fs_splits.y_train)
    return regr


@memory.cache
def fit_rf_out_season(**rf_params):
    regr = RandomForestRegressor(**rf_params)
    # Make sure all cores are used.
    regr.n_jobs = get_ncpus()
    regr.fit(out_fs_splits.X_train, out_fs_splits.y_train)
    return regr


@memory.cache
def fit_all(**rf_params):
    regr = RandomForestRegressor(**rf_params)
    # Make sure all cores are used.
    regr.n_jobs = get_ncpus()
    regr.fit(all_splits.X_train, all_splits.y_train)
    return regr


logger.info("Fitting models.")

for fit_func, split_data, raw_data, name in zip(
    tqdm((fit_rf_in_season, fit_rf_out_season, fit_all)),
    (in_fs_splits, out_fs_splits, all_splits),
    (in_fire_season, out_fire_season, all_season),
    ("in_season", "out_season", "all_season"),
):
    regr = fit_func(**rf_params)

    # Reset n_jobs in case a cached result with settings was loaded.
    regr.n_jobs = get_ncpus()

    y_pred = regr.predict(split_data.X_test)

    # Carry out predictions on the training dataset to diagnose overfitting.
    y_pred_train = regr.predict(split_data.X_train)

    results = {}
    results["R2_train"] = regr.score(split_data.X_train, split_data.y_train)
    results["R2_test"] = regr.score(split_data.X_test, split_data.y_test)

    model_name = "RF"
    print(f"{model_name} R2 train: {results['R2_train']}")
    print(f"{model_name} R2 test: {results['R2_test']}")

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_], axis=0)

    importances_df = pd.DataFrame(
        {
            "Name": raw_data.exog_data.columns.values,
            "Importance": importances,
            "Importance STD": std,
            "Ratio": np.array(std) / np.array(importances),
        }
    ).sort_values("Importance", ascending=False)
    print(
        "\n"
        + str(
            importances_df.to_string(
                index=False, float_format="{:0.3f}".format, line_width=200
            )
        )
    )
    importances_df.to_csv(
        os.path.join(FigureSaver.directory, f"{name}_importances.csv"), index=False
    )

    # logger.info("Plotting PDPs.")
    #
    # with FigureSaver("pdp"):
    #     fig_axes = partial_dependence_plot(
    #         regr,
    #         raw_data.exog_data.loc,
    #         split_data.X_test.columns,
    #         grid_resolution=60,
    #         coverage=0.3,
    #         predicted_name="Burned Area",
    #         single_plots=False,
    #         log_x_scale=("Dry Day Period", "popd"),
    #         plot_range=False,
    #     )
