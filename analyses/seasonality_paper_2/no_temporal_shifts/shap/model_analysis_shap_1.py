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


# About 2 s / sample
# Expect ~ 1 hr per job -> 2000 samples in 2 hrs each (allowing for poor performance)


def func():
    # Used to re-compute specific failed jobs, `None` otherwise.
    indices = [
        8,
        50,
        51,
        58,
        60,
        61,
        62,
        63,
        64,
        68,
        94,
        219,
        265,
        266,
        268,
        314,
        315,
        317,
        319,
        320,
        321,
        322,
        347,
        348,
        379,
        383,
        384,
        392,
        419,
        420,
        421,
        422,
        453,
        456,
        458,
        472,
        473,
        474,
        475,
        476,
        489,
        537,
        538,
        540,
        542,
        571,
        572,
        573,
        574,
        654,
        655,
        664,
        665,
        666,
    ]

    index = int(os.environ["PBS_ARRAY_INDEX"])

    if indices is not None:
        index = indices[index]

    print("Index:", index)

    X_train, X_test, y_train, y_test = data_split_cache.load()
    rf = get_model()

    job_samples = 2000

    tree_path_dependent_shap_cache = SimpleCache(
        f"tree_path_dependent_shap_{index}_{job_samples}",
        cache_dir=os.path.join(CACHE_DIR, "shap"),
    )

    @tree_path_dependent_shap_cache
    def cached_get_shap_values(model, X):
        return get_shap_values(model, X, interaction=False)

    cached_get_shap_values(rf, X_train[index * job_samples : (index + 1) * job_samples])


if __name__ == "__main__":
    handle_array_job_args(
        Path(__file__).resolve(),
        func,
        ncpus=1,
        mem="7gb",
        walltime="04:00:00",
        max_index=53,
    )
