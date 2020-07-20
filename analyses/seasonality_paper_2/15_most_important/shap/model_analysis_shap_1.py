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
        2,
        11,
        12,
        13,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        202,
        203,
        205,
        206,
        209,
        210,
        211,
        212,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        235,
        236,
        237,
        238,
        242,
        243,
        250,
        251,
        254,
        256,
        265,
        277,
        307,
        387,
        388,
        414,
        415,
        427,
        428,
        429,
        432,
        433,
        447,
        488,
        489,
        494,
        550,
        557,
        573,
        574,
        585,
        633,
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
        max_index=71,
    )
