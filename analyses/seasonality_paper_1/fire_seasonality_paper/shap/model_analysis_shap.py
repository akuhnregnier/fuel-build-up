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


# About 2 s / sample
# Expect ~ 1 hr per job -> 2000 samples in 2 hrs each (allowing for poor performance)


def func():
    # Used to re-compute specific failed jobs, `None` otherwise.
    indices = [
        14,
        43,
        45,
        59,
        62,
        73,
        79,
        94,
        104,
        110,
        122,
        140,
        142,
        150,
        157,
        161,
        187,
        196,
        232,
        236,
        247,
        264,
        298,
        306,
        311,
        312,
        398,
        402,
        408,
        410,
        446,
        454,
        456,
        459,
        460,
        461,
        462,
        464,
        467,
        469,
        470,
        471,
        482,
        483,
        484,
        492,
        493,
        494,
        495,
        496,
        497,
        516,
        517,
        518,
        523,
        535,
        537,
        543,
        554,
        557,
        565,
        572,
        573,
        644,
        645,
        646,
        647,
        648,
        671,
        673,
        674,
        676,
        677,
        678,
        679,
        700,
        701,
        715,
        750,
        751,
        752,
        753,
        803,
        804,
        805,
        814,
        816,
        817,
        820,
        822,
        827,
        829,
        830,
        831,
        952,
        953,
        954,
        989,
    ]

    index = int(os.environ["PBS_ARRAY_INDEX"])

    if indices is not None:
        index = indices[index]

    print("Index:", index)

    X_train, X_test, y_train, y_test = data_split_cache.load()
    results, rf = cross_val_cache.load()

    job_samples = 2000

    tree_path_dependent_shap_cache = SimpleCache(
        f"tree_path_dependent_shap_{index}_{job_samples}",
        cache_dir=os.path.join(CACHE_DIR, "shap"),
    )

    @tree_path_dependent_shap_cache
    def get_interact_shap_values(model, X):
        return get_shap_values(model, X, interaction=False)

    get_interact_shap_values(
        rf, X_train[index * job_samples : (index + 1) * job_samples]
    )


if __name__ == "__main__":
    handle_array_job_args(
        Path(__file__).resolve(),
        func,
        ncpus=1,
        mem="5gb",
        walltime="03:00:00",
        max_index=100,
    )
