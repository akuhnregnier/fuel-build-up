#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

from wildfires.qstat import get_ncpus
from wildfires.utils import handle_array_job_args

try:
    # This will only work after the path modification carried out in the job script.
    from specific import (
        combinations,
        cross_val_cache,
        data_split_cache,
        figure_saver,
        parallel_backend,
        pdp,
    )
except ImportError:
    """Not running as an HPC job yet."""


def func():
    def save_pdp_plot_2d(model, X_train, features, n_jobs):
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
        figure_saver.save_figure(fig, "__".join(features), sub_directory="pdp_2d")

    X_train, X_test, y_train, y_test = data_split_cache.load()
    results, rf = cross_val_cache.load()
    columns_list = list(combinations(X_train.columns, 2))

    index = int(os.environ["PBS_ARRAY_INDEX"])
    print("Index:", index)
    print("Columns:", columns_list[index])

    ncpus = get_ncpus()
    print("NCPUS:", ncpus)

    # Use the array index to select the desired columns.
    save_pdp_plot_2d(rf, X_train, columns_list[index], ncpus)


if __name__ == "__main__":
    handle_array_job_args(
        Path(__file__).resolve(),
        func,
        ncpus=7,
        mem="60gb",
        walltime="24:00:00",
        max_index=104,
    )
