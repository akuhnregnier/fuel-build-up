# -*- coding: utf-8 -*-
"""Investigate how the threshold affects the estimates.

The threshold is a fraction, relative to the maximum BA at each location.

"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from wildfires.analysis import FigureSaver, cube_plotting, thres_fire_season_stats
from wildfires.dask_cx1 import get_parallel_backend
from wildfires.logging_config import enable_logging


def outputs_plotting(thres, outputs):
    """Plotting of fire season statistics.

    Args:
        thres (float): Threshold used to generate the data.
        outputs: Output of `wildfires.analysis.thres_fire_season_stats`.

    """
    enable_logging()
    FigureSaver.debug = True
    FigureSaver.directory = os.path.join(os.path.expanduser("~"), "tmp", "fire_season")
    os.makedirs(FigureSaver.directory, exist_ok=True)

    for dataset_outputs in outputs:
        name = dataset_outputs[0]
        starts = dataset_outputs[1]
        ends = dataset_outputs[2]
        sizes = dataset_outputs[3]
        fractions = dataset_outputs[5]

        for plot_type, data, cmap, boundaries in zip(
            ("start (month)", "end (month)", "length (months)", "fraction (1)"),
            (starts, ends, sizes, fractions),
            (*("twilight",) * 2, *("brewer_RdYlBu_11_r",) * 2),
            (*(np.arange(0, 12),) * 3, None),
        ):
            with FigureSaver(
                f"{name}_thres_{str(thres).replace('.', '_')}_{plot_type}"
            ):
                mpl.rc("figure", figsize=(7.4, 3.3))
                cube_plotting(
                    data,
                    coastline_kwargs=dict(linewidth=0.5),
                    cmap=cmap,
                    label=plot_type,
                    title=name,
                    boundaries=boundaries,
                )
            # Close all figures after saving.
            plt.close("all")


if __name__ == "__main__":
    plt.close("all")
    enable_logging()

    thresholds = np.round(np.geomspace(1e-4, 1e-1, 10), 5)

    with get_parallel_backend(cores=2, memory="8GB", walltime="00:30:00"):
        # Generate data.
        outputs_list = Parallel(verbose=10)(
            delayed(thres_fire_season_stats)(thres) for thres in thresholds
        )

        # Plot data.
        Parallel(verbose=10)(
            delayed(outputs_plotting)(thres, outputs)
            for thres, outputs in zip(thresholds, outputs_list)
        )
