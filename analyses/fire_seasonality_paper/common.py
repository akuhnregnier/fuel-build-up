# -*- coding: utf-8 -*-
import logging
import os
import re
import sys
import warnings

import matplotlib as mpl
from loguru import logger as loguru_logger

from wildfires.analysis import FigureSaver, data_processing
from wildfires.dask_cx1 import (
    DaskRandomForestRegressor,
    fit_dask_rf_grid_search_cv,
    get_client,
)
from wildfires.data import (
    DATA_DIR,
    HYDE,
    VODCA,
    WWLLN,
    AvitabileThurnerAGB,
    Copernicus_SWI,
    Datasets,
    ERA5_DryDayPeriod,
    ERA5_Temperature,
    ESA_CCI_Landcover_PFT,
    GFEDv4,
    GlobFluo_SIF,
    MOD15A2H_LAI_fPAR,
    get_memory,
)
from wildfires.joblib.cloudpickle_backend import register_backend as register_cl_backend
from wildfires.logging_config import enable_logging

loguru_logger.enable("alepython")
loguru_logger.remove()
loguru_logger.add(sys.stderr, level="WARNING")

logger = logging.getLogger(__name__)
enable_logging("jupyter")

warnings.filterwarnings("ignore", ".*Collapsing a non-contiguous coordinate.*")
warnings.filterwarnings("ignore", ".*DEFAULT_SPHERICAL_EARTH_RADIUS*")
warnings.filterwarnings("ignore", ".*guessing contiguous bounds*")

normal_coast_linewidth = 0.5
mpl.rc("figure", figsize=(14, 6))
mpl.rc("font", size=9.0)

save_name = "fire_seasonality_paper"

figure_saver = FigureSaver(directories=os.path.join("~", "tmp", save_name), debug=True)
memory = get_memory(save_name, verbose=100)
CACHE_DIR = os.path.join(DATA_DIR, ".pickle", save_name)

register_cl_backend()
data_memory = get_memory("analysis_lags_rf_cross_val", backend="cloudpickle", verbose=2)


# Creating the Data Structures used for Fitting
@data_memory.cache
def get_data(
    shift_months=[1, 3, 6, 9, 12, 18, 24], selection_variables=None, masks=None
):
    target_variable = "GFED4 BA"

    # Variables required for the above.
    required_variables = [target_variable]

    # Dataset selection.

    selection_datasets = [
        AvitabileThurnerAGB(),
        Copernicus_SWI(),
        ERA5_Temperature(),
        ESA_CCI_Landcover_PFT(),
        GFEDv4(),
        HYDE(),
        WWLLN(),
    ]
    # These datasets will potentially be shifted.
    datasets_to_shift = [
        ERA5_DryDayPeriod(),
        MOD15A2H_LAI_fPAR(),
        VODCA(),
        GlobFluo_SIF(),
    ]
    selection_datasets += datasets_to_shift
    if shift_months is not None:
        for shift in shift_months:
            for shift_dataset in datasets_to_shift:
                selection_datasets.append(
                    shift_dataset.get_temporally_shifted_dataset(
                        months=-shift, deep=False
                    )
                )

    if selection_variables is None:
        selection_variables = [
            "AGB Tree",
            "Diurnal Temp Range",
            "Dry Day Period",
            "FAPAR",
            "LAI",
            "Max Temp",
            "SIF",
            "SWI(1)",
            "ShrubAll",
            "TreeAll",
            "VOD Ku-band",
            "lightning",
            "pftCrop",
            "pftHerb",
            "popd",
        ]
        if shift_months is not None:
            for shift in shift_months:
                selection_variables.extend(
                    [
                        f"LAI {-shift} Month",
                        f"FAPAR {-shift} Month",
                        f"Dry Day Period {-shift} Month",
                        f"VOD Ku-band {-shift} Month",
                        f"SIF {-shift} Month",
                    ]
                )

    selection_variables = list(set(selection_variables).union(required_variables))

    selection = Datasets(selection_datasets).select_variables(selection_variables)
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = data_processing(
        selection,
        which="climatology",
        transformations={},
        deletions=[],
        use_lat_mask=False,
        use_fire_mask=False,
        target_variable=target_variable,
        masks=masks,
    )
    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


@data_memory.cache
def get_offset_data(
    shift_months=[1, 3, 6, 9, 12, 18, 24], selection_variables=None, masks=None
):
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = get_data(
        shift_months=shift_months, selection_variables=selection_variables, masks=masks
    )

    to_delete = []

    for column in exog_data:
        match = re.search(r"-\d{1,2}", column)
        if match:
            span = match.span()
            # Change the string to reflect the shift.
            original_offset = int(column[slice(*span)])
            if original_offset > -12:
                # Only shift months that are 12 or more months before the current month.
                continue
            comp = -(-original_offset % 12)
            new_column = " ".join(
                (
                    column[: span[0] - 1],
                    f"{original_offset} - {comp}",
                    column[span[1] + 1 :],
                )
            )
            if comp == 0:
                comp_column = column[: span[0] - 1]
            else:
                comp_column = " ".join(
                    (column[: span[0] - 1], f"{comp}", column[span[1] + 1 :])
                )
            print(column, comp_column)
            exog_data[new_column] = exog_data[column] - exog_data[comp_column]
            to_delete.append(column)

    for column in to_delete:
        del exog_data[column]

    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )
