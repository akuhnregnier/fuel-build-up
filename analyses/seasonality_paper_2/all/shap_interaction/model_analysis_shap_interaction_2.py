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
        get_model,
        data_split_cache,
        get_shap_values,
    )
except ImportError:
    """Not running as an HPC job yet."""


def func():
    # Used to re-compute specific failed jobs, `None` otherwise.
    indices = [
        108,
        148,
        153,
        454,
        515,
        532,
        551,
        581,
        603,
        607,
        608,
        618,
        646,
        647,
        648,
        669,
        850,
        852,
        867,
        944,
        945,
        983,
        984,
        985,
        987,
        989,
        991,
        1006,
        1009,
        1010,
        1020,
        1072,
        1182,
        1205,
        1211,
        1215,
        1237,
        1262,
        1263,
        1271,
        1295,
        1296,
        1310,
        1314,
        1315,
        1324,
        1339,
        1345,
        1390,
        1405,
        1521,
        1546,
        1555,
        1582,
        1671,
        1835,
        2182,
        2399,
        2826,
        2918,
        2971,
        2972,
        3006,
        3061,
        3092,
        3093,
        3211,
        3227,
        3233,
        3297,
        3308,
        3396,
        3397,
        4125,
        4148,
        4177,
        4203,
        4312,
        4339,
        4340,
        4365,
        4366,
        4371,
        4406,
        4407,
        4408,
        4430,
        4431,
        4483,
        4484,
        4486,
        4546,
        4559,
        4713,
        4994,
        4996,
        5940,
        5965,
    ]

    index = int(os.environ["PBS_ARRAY_INDEX"])

    if indices is not None:
        index = indices[index]

    print("Index:", index)

    X_train, X_test, y_train, y_test = data_split_cache.load()
    rf = get_model()

    job_samples = 50

    tree_path_dependent_shap_interact_cache = SimpleCache(
        f"tree_path_dependent_shap_interact_{index}_{job_samples}",
        cache_dir=os.path.join(CACHE_DIR, "shap_interaction"),
    )

    @tree_path_dependent_shap_interact_cache
    def cached_get_interact_shap_values(model, X):
        return get_shap_values(model, X, interaction=True)

    cached_get_interact_shap_values(
        rf, X_train[index * job_samples : (index + 1) * job_samples]
    )


if __name__ == "__main__":
    handle_array_job_args(
        Path(__file__).resolve(),
        func,
        ncpus=1,
        mem="7gb",
        walltime="12:00:00",
        max_index=97,
    )
