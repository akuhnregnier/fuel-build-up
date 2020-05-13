# -*- coding: utf-8 -*-
"""Since models have preferred the 24-month to the 12-month shifted variable,
investigate the correlation between variables at 12-month relative shifts.

"""
import logging
import os
import warnings

import matplotlib as mpl
import numpy as np
import pandas as pd

from wildfires.analysis import *
from wildfires.data import *
from wildfires.joblib.cloudpickle_backend import register_backend
from wildfires.logging_config import enable_logging
from wildfires.utils import *

logger = logging.getLogger(__name__)
enable_logging()

register_backend()

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds*")

memory = get_memory(
    "analysis_correlation_12_month_shift", backend="cloudpickle", verbose=100
)

FigureSaver.debug = True
FigureSaver.directory = os.path.expanduser(
    os.path.join("~", "tmp", "correlation_12_month_shift")
)
os.makedirs(FigureSaver.directory, exist_ok=True)

normal_coast_linewidth = 0.5
mpl.rc("figure", figsize=(14, 6))
mpl.rc("font", size=9.0)

np.random.seed(1)


@memory.cache
def get_data(shift_months=(12, 24)):
    """

    Returns:
        iterable of wildfires.data.Datasets: monthly, mean, climatology.

    """
    selection_datasets = [
        Copernicus_SWI(),
        ERA5_CAPEPrecip(),
        ERA5_DryDayPeriod(),
        GlobFluo_SIF(),
        MOD15A2H_LAI_fPAR(),
        VODCA(),
    ]
    shifted_datasets = []
    for shift in shift_months:
        for shift_dataset in selection_datasets:
            shifted_datasets.append(
                shift_dataset.get_temporally_shifted_dataset(months=-shift)
            )
    shifted_datasets.extend(selection_datasets)
    shifted_datasets = Datasets(shifted_datasets)

    selected_variables = [
        "SWI(1)",
        "CAPE x Precip",
        "Dry Day Period",
        "SIF",
        "FAPAR",
        "LAI",
        "VOD Ku-band",
    ]
    shifted_selected_variables = []
    for shift in shift_months:
        for var_name in selected_variables:
            shifted_selected_variables.append(f"{var_name} {-shift} Month")
    shifted_selected_variables.extend(selected_variables)

    shifted_datasets.select_variables(shifted_selected_variables)

    prepared_datsets = prepare_selection(shifted_datasets)
    for dataset in prepared_datsets:
        dataset.cubes.realise_data()
    return prepared_datsets


@memory.cache
def processed_data(shift_months=(12, 24)):
    """

    Returns:
        iterable of pandas.DataFrame: monthly_exog_data, mean_exog_data,
            climatology_exog_data.

    """
    exog_datas = []
    for datasets in get_data():
        logger.info("Homogenising masks.")
        datasets.homogenise_masks()

        land_mask = ~get_land_mask(
            n_lon=datasets.cubes[0].coord("longitude").points.shape[0]
        )

        logger.info("Extra processing for 'SIF'.")
        for name in datasets.pretty_variable_names:
            if "SIF" not in name:
                continue
            sif_cube = datasets.select_variables(name, inplace=False).cube
            invalid_mask = np.logical_or(
                sif_cube.data.data > 20, sif_cube.data.data < 0
            )
            logger.info(f"Masking {np.sum(invalid_mask)} invalid values for {name}.")
            sif_cube.data.mask |= invalid_mask

        logger.info("Making masks uniform.")
        master_mask = datasets.cubes[0].data.mask
        for cube in datasets.cubes[1:]:
            master_mask |= cube.data.mask

        masks_to_apply = [master_mask, land_mask]
        logger.info("Applying masks.")
        datasets.apply_masks(*masks_to_apply)

        data = []
        for cube in datasets.cubes:
            data.append(get_unmasked(cube.data).reshape(-1, 1))

        exog_datas.append(
            pd.DataFrame(np.hstack(data), columns=datasets.pretty_variable_names)
        )
    return exog_datas


exog_datas = processed_data()

for sampling, exog_data in zip(
    ("monthly", "climatology"), (exog_datas[0], exog_datas[2])
):
    selected_variables = [
        "SWI(1)",
        "CAPE x Precip",
        "Dry Day Period",
        "SIF",
        "FAPAR",
        "LAI",
        "VOD Ku-band",
    ]
    for variable in selected_variables:
        columns = [column for column in exog_data.columns if variable in column]
        with FigureSaver(f"{variable}_{sampling}_yearly_shift_correlations"):
            corr_plot(exog_data[columns])

        print(f"Correlations: {sampling} {variable}")
        print(exog_data[columns].corr())
