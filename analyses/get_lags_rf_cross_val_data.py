# -*- coding: utf-8 -*-
import logging

from wildfires.analysis import *
from wildfires.data import *
from wildfires.joblib.cloudpickle_backend import register_backend

logger = logging.getLogger(__name__)

register_backend()
memory = get_memory("analysis_lags_rf_cross_val", backend="cloudpickle", verbose=2)

# Creating the Data Structures used for Fitting


@memory.cache
def get_data(shift_months=None, selection_variables=None, masks=None):
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
