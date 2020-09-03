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
        54,
        55,
        56,
        57,
        58,
        65,
        66,
        72,
        73,
        82,
        83,
        84,
        85,
        92,
        93,
        94,
        112,
        113,
        114,
        115,
        116,
        118,
        119,
        128,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        143,
        144,
        147,
        148,
        164,
        165,
        166,
        167,
        172,
        173,
        174,
        196,
        197,
        198,
        199,
        201,
        203,
        204,
        212,
        215,
        216,
        217,
        218,
        220,
        221,
        222,
        223,
        224,
        230,
        231,
        236,
        237,
        247,
        248,
        249,
        250,
        251,
        254,
        255,
        256,
        259,
        293,
        294,
        295,
        296,
        298,
        299,
        300,
        305,
        307,
        308,
        309,
        310,
        311,
        312,
        313,
        314,
        315,
        325,
        326,
        328,
        329,
        352,
        353,
        354,
        355,
        356,
        360,
        361,
        362,
        370,
        419,
        420,
        421,
        422,
        425,
        426,
        427,
        428,
        431,
        432,
        433,
        434,
        435,
        436,
        437,
        438,
        439,
        446,
        447,
        455,
        456,
        489,
        490,
        491,
        492,
        493,
        495,
        496,
        497,
        504,
        532,
        533,
        534,
        535,
        538,
        539,
        540,
        546,
        550,
        551,
        552,
        556,
        557,
        558,
        559,
        560,
        561,
        577,
        578,
        587,
        588,
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
        walltime="05:00:00",
        max_index=154,
    )
