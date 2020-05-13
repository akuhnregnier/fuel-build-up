# -*- coding: utf-8 -*-
"""Investigate how fire season estimates differ between datasets.

"""
import logging
import math
import os
import warnings

import iris
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend

from wildfires.analysis import *
from wildfires.data import *
from wildfires.logging_config import enable_logging
from wildfires.qstat import get_ncpus
from wildfires.utils import *

logger = logging.getLogger(__name__)
enable_logging()

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds*")

memory = get_memory("analysis_fire_season_dataset_diffs", verbose=100)

FigureSaver.debug = True
FigureSaver.directory = os.path.expanduser(
    os.path.join("~", "tmp", "fire_season_dataset_diffs")
)
os.makedirs(FigureSaver.directory, exist_ok=True)

normal_coast_linewidth = 0.5
mpl.rc("figure", figsize=(14, 6))
mpl.rc("font", size=9.0)

np.random.seed(1)

n_jobs = 5
with parallel_backend(
    "loky", n_jobs=n_jobs, inner_max_num_threads=math.floor(get_ncpus() / n_jobs)
):
    outputs = thres_fire_season_stats(0.1)


dataset_names = [output[0] for output in outputs]
lengths = [output[3].reshape(1, *output[3].shape) for output in outputs]

# Stack the lengths into one array.
lengths = np.ma.vstack(lengths)

mean_length = np.ma.mean(lengths, axis=0)

# Mean BAs
ba_variable_names = (
    "CCI MERIS BA",
    "CCI MODIS BA",
    "GFED4 BA",
    "GFED4s BA",
    "MCD64CMQ BA",
)
mean_ba_cubes = (
    prepare_selection(
        Datasets([globals()[name]() for name in dataset_names]), which="mean"
    )
    .select_variables(ba_variable_names)
    .cubes
)

mean_bas = []
for mean_ba_cube in mean_ba_cubes:
    mean_bas.append(
        mean_ba_cube.collapsed(
            ("latitude", "longitude"),
            iris.analysis.MEAN,
            weights=iris.analysis.cartography.area_weights(mean_ba_cube),
        ).data
    )

# Diffs from the mean.
deviations_cube = dummy_lat_lon_cube(lengths - mean_length.reshape(1, 720, 1440))
deviations = deviations_cube.collapsed(
    ["latitude", "longitude"],
    iris.analysis.MEAN,
    weights=iris.analysis.cartography.area_weights(deviations_cube),
).data
deviation_df = pd.DataFrame(
    [
        (name, deviation, mean_ba)
        for name, deviation, mean_ba in zip(dataset_names, deviations, mean_bas)
    ],
    columns=("Name", "Deviation from Mean", "Mean BA"),
).sort_values("Deviation from Mean", ascending=False)

print(
    deviation_df.to_string(index=False, float_format="{:0.3f}".format, line_width=200)
)

deviation_df.to_csv(
    os.path.join(FigureSaver.directory, f"season_length_mean_deviations.csv"),
    index=False,
)


with FigureSaver("mean_length"):
    cube_plotting(
        mean_length,
        coastline_kwargs=dict(linewidth=0.5),
        cmap="brewer_RdYlBu_11_r",
        label="length (months)",
        title="Mean Length",
        boundaries=np.arange(0, 12),
    )

std_length = np.ma.std(lengths, axis=0)

with FigureSaver("std_length"):
    cube_plotting(
        std_length,
        coastline_kwargs=dict(linewidth=0.5),
        cmap="inferno",
        label="length (months)",
        title="Std Length (Between Datasets)",
    )

mean_ba = prepare_selection(Datasets(GFEDv4s()), which="mean").cube

with FigureSaver("std_length_corr_mean_ba_gfedv4s"):

    combined_mask = mean_ba.data.mask | std_length.mask
    mean_ba.data.mask = combined_mask
    std_length.mask = combined_mask

    plt.figure()
    plt.hexbin(np.log(get_unmasked(mean_ba.data)), get_unmasked(std_length), bins="log")
    plt.xlabel("log(Mean BA GFEDv4s)")
    plt.ylabel("STD Fire Season Length")

    # plt.xscale('log')
    plt.show()

with FigureSaver("gfedv4s_length_deviation"):
    cube_plotting(
        lengths[dataset_names.index("GFEDv4s")] - mean_length,
        coastline_kwargs=dict(linewidth=0.5),
        cmap="inferno",
        label="length (months)",
        title="GFED4s Fire Season Length - Mean",
    )
