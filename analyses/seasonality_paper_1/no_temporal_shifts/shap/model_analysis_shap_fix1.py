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
        3,
        4,
        154,
        155,
        211,
        240,
        251,
        289,
        293,
        294,
        330,
        331,
        334,
        345,
        380,
        381,
        392,
        410,
        421,
        422,
        423,
        439,
        460,
        461,
        467,
        468,
        483,
        503,
        516,
        522,
        523,
        528,
        536,
        539,
        540,
        541,
        542,
        543,
        558,
        560,
        564,
        567,
        568,
        580,
        585,
        586,
        590,
        591,
        617,
        618,
        619,
        620,
        625,
        626,
        633,
        658,
        659,
        660,
        661,
        662,
        663,
        668,
        687,
        688,
        689,
        691,
        692,
        693,
        694,
        695,
        696,
        697,
        698,
        700,
        702,
        703,
        721,
        746,
        754,
        755,
        762,
        763,
        790,
        791,
        792,
        793,
        794,
        798,
        806,
        814,
        818,
        827,
        828,
        829,
        832,
        844,
        845,
        861,
        862,
        863,
        864,
        866,
        884,
        888,
        889,
        890,
        891,
        892,
        897,
        900,
        902,
        903,
        904,
        905,
        906,
        910,
        911,
        912,
        913,
        927,
        928,
        929,
        930,
        937,
        938,
        948,
        949,
        950,
        952,
        953,
        956,
        960,
        962,
        974,
        975,
        976,
        977,
        978,
        980,
        982,
        983,
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
        walltime="07:00:00",
        max_index=140,
    )
