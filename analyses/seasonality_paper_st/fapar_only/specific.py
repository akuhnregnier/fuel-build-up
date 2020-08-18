# -*- coding: utf-8 -*-
import sys
import warnings
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.utils.deprecation"
)
from common import *

warnings.filterwarnings(
    "always", category=FutureWarning, module="sklearn.utils.deprecation"
)

figure_saver = FigureSaver(
    directories=Path("~") / "tmp" / PROJECT_DIR.parent.name / PROJECT_DIR.name,
    debug=True,
)
map_figure_saver = figure_saver(**map_figure_saver_kwargs)

memory = get_memory("__".join((PROJECT_DIR.parent.name, PROJECT_DIR.name)), verbose=100)
CACHE_DIR = Path(DATA_DIR) / ".pickle" / PROJECT_DIR.parent.name / PROJECT_DIR.name

data_split_cache = SimpleCache("data_split", cache_dir=CACHE_DIR)

save_ale_2d_and_get_importance = partial(
    save_ale_2d_and_get_importance, figure_saver=figure_saver
)
save_pdp_plot_2d = partial(save_pdp_plot_2d, figure_saver=figure_saver)
save_ale_plot_1d_with_ptp = partial(
    save_ale_plot_1d_with_ptp, figure_saver=figure_saver
)
save_pdp_plot_1d = partial(
    save_pdp_plot_1d, CACHE_DIR=CACHE_DIR, figure_saver=figure_saver
)
multi_ale_plot_1d = partial(multi_ale_plot_1d, figure_saver=figure_saver)

# Number of SHAP jobs.
try:
    X_train, X_test, y_train, y_test = data_split_cache.load()
    # Maximum job array index (inclusive).
    shap_params["max_index"] = math.floor(X_train.shape[0] / shap_params["job_samples"])
    # Upper bound only.
    shap_params["total_samples"] = (shap_params["max_index"] + 1) * shap_params[
        "job_samples"
    ]
except NoCachedDataError:
    warnings.warn(
        "Processed data not found, not calculating 'max_index' or 'total_samples'."
    )

# Upper bound only.
shap_interact_params["total_samples"] = (
    shap_interact_params["max_index"] + 1
) * shap_interact_params["job_samples"]

# SHAP cache.
shap_cache = SimpleCache("shap_cache", cache_dir=CACHE_DIR / Path("shap"))
shap_interact_cache = SimpleCache(
    "shap_interact_cache", cache_dir=CACHE_DIR / Path("shap_interaction")
)

interact_data_cache = SimpleCache("SHAP_interact_data", cache_dir=CACHE_DIR)


# Redefine the common functionality for our use-case - no shifted variables.
_common_get_data = get_data
_common_get_offset_data = get_offset_data

selected_features = (
    "Dry Day Period",
    "Max Temp",
    "pftCrop",
    "popd",
    "Diurnal Temp Range",
    "Dry Day Period -3 Month",
    "AGB Tree",
    "Dry Day Period -1 Month",
    "SWI(1) 3NN",
    "Dry Day Period -9 Month",
    f"FAPAR {n_months}NN",
    f"FAPAR {n_months}NN -1 Month",
    f"FAPAR {n_months}NN -3 Month",
    f"FAPAR {n_months}NN -6 Month",
    f"FAPAR {n_months}NN -9 Month",
)

offset_selected_features = []
for column in selected_features:
    match = re.search(r"-\d{1,2}", column)
    if match:
        span = match.span()
        # Change the string to reflect the shift.
        original_offset = int(column[slice(*span)])
        if original_offset > -12:
            # Only shift months that are 12 or more months before the current month.
            offset_selected_features.append(column)
            continue
        comp = -(-original_offset % 12)
        new_column = " ".join(
            (
                column[: span[0] - 1],
                f"{original_offset} - {comp}",
                column[span[1] + 1 :],
            )
        )
        offset_selected_features.append(new_column)
    else:
        offset_selected_features.append(column)


@wraps(_common_get_data)
def get_data(*args, **kwargs):
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = _common_get_data(*args, **kwargs)

    # We need to subset exog_data, filled_datasets, and masked_datasets.
    exog_data = exog_data[list(selected_features)]
    filled_datasets = filled_datasets.select_variables(selected_features)
    masked_datasets = masked_datasets.select_variables(selected_features)

    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


@wraps(_common_get_offset_data)
def get_offset_data(*args, **kwargs):
    (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    ) = _common_get_offset_data(*args, **kwargs)

    # We need to subset exog_data, filled_datasets, and masked_datasets.
    exog_data = exog_data[list(offset_selected_features)]
    filled_datasets = filled_datasets.select_variables(selected_features)
    masked_datasets = masked_datasets.select_variables(selected_features)

    return (
        endog_data,
        exog_data,
        master_mask,
        filled_datasets,
        masked_datasets,
        land_mask,
    )


def get_model(X_train=None, y_train=None):
    return common_get_model(cache_dir=CACHE_DIR, X_train=X_train, y_train=y_train)
