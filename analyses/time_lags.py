#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import warnings
from itertools import product

import matplotlib as mpl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

import wildfires.analysis
from wildfires.analysis import *
from wildfires.dask_cx1 import get_client
from wildfires.data import *
from wildfires.logging_config import enable_logging

FigureSaver.debug = True
FigureSaver.directory = os.path.expanduser(os.path.join("~", "tmp", "time_lags"))
os.makedirs(FigureSaver.directory, exist_ok=True)
logger = logging.getLogger(__name__)

enable_logging()

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds*")

normal_coast_linewidth = 0.5
mpl.rc("figure", figsize=(14, 6))
mpl.rc("font", size=9.0)

np.random.seed(1)

memory = get_memory("analysis_time_lags", verbose=100)


# Creating the Data Structures used for Fitting

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

(
    s_endog_data,
    s_exog_data,
    s_master_mask,
    s_filled_datasets,
    s_masked_datasets,
    s_land_mask,
) = wildfires.analysis.time_lags.get_data(
    shift_months=[1, 3, 6, 12, 24], selection_variables=selection_variables
)

(
    e_s_endog_data,
    e_s_exog_data,
    e_s_master_mask,
    e_s_filled_datasets,
    e_s_masked_datasets,
    e_s_land_mask,
) = wildfires.analysis.time_lags.get_data(
    shift_months=[1, 3, 6, 12, 24], selection_variables=ext_selection_variables
)

# Hyperparameter Optimisation

# Define the training and test data.
X_train, X_test, y_train, y_test = train_test_split(
    s_exog_data, s_endog_data, random_state=1, shuffle=True, test_size=0.3
)

# Worker specifications.
specs = {"memory": "2GB", "walltime": "01:00:00", "cores": 4}
# Connect to an existing cluster with at least those specs.
client = get_client(**specs)

# Define the parameter space.
parameters_RF = {
    "n_estimators": [10, 50, 100],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [3, 10, 20],
    "max_features": ["auto"],
    "bootstrap": [False, True],
    "random_state": [1],
}


def fit_func(X, y, rf_params):
    rf = RandomForestRegressor(**rf_params)
    scores = cross_val_score(rf, X, y, cv=5)
    # XXX: What about the n_jobs parameters for the above two things?
    # Optionally fit model on all the data and store the fitted model using pickle.
    return scores


parameter_grid = list(
    dict(zip(parameters_RF, params))
    for params in product(*list(parameters_RF.values()))
)

logger.info("Scattering training data.")

X_train_fut = client.scatter(X_train, broadcast=True)
y_train_fut = client.scatter(y_train, broadcast=True)

logger.info("Submitting tasks.")

score_futures = []
for single_parameters in parameter_grid:
    score_futures.append(
        client.submit(fit_func, X_train_fut, y_train_fut, single_parameters)
    )

logger.info("Waiting for tasks to finish.")
scores_list = client.gather(score_futures)
