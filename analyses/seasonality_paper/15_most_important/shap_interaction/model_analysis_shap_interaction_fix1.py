#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

from wildfires.utils import handle_array_job_args

try:
    # This will only work after the path modification carried out in the job script.
    from specific import (
        CACHE_DIR,
        SimpleCache,
        cross_val_cache,
        data_split_cache,
        get_shap_values,
    )
except ImportError:
    """Not running as an HPC job yet."""


def func():
    # Used to re-compute specific failed jobs, `None` otherwise.
    indices = [
        115,
        989,
        1102,
        1163,
        1236,
        1275,
        1276,
        1277,
        1377,
        1378,
        1514,
        1515,
        1516,
        1716,
        1771,
        1772,
        1773,
        1774,
        1867,
        1940,
        1955,
        2095,
        2174,
        2301,
        2827,
        2908,
        2947,
        2958,
        2960,
        3138,
        3168,
        3169,
        3782,
        3783,
        3784,
        3785,
        3786,
        3787,
        4108,
        4128,
        4129,
        4130,
        4131,
        4132,
        4224,
        4281,
        4302,
        4357,
        4435,
        4436,
        4579,
        4673,
        4865,
        4904,
    ]

    index = int(os.environ["PBS_ARRAY_INDEX"])

    if indices is not None:
        index = indices[index]

    print("Index:", index)

    X_train, X_test, y_train, y_test = data_split_cache.load()
    results, rf = cross_val_cache.load()

    job_samples = 50

    tree_path_dependent_shap_interact_cache = SimpleCache(
        f"tree_path_dependent_shap_interact_{index}_{job_samples}",
        cache_dir=os.path.join(CACHE_DIR, "shap_interaction"),
    )

    @tree_path_dependent_shap_interact_cache
    def get_interact_shap_values(model, X):
        return get_shap_values(model, X, interaction=True)

    get_interact_shap_values(
        rf, X_train[index * job_samples : (index + 1) * job_samples]
    )


if __name__ == "__main__":
    handle_array_job_args(
        Path(__file__).resolve(),
        func,
        ncpus=1,
        mem="6gb",
        walltime="10:00:00",
        max_index=53,
    )
