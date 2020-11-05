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

figure_saver = PaperFigureSaver(
    directories=Path("~") / "tmp" / PROJECT_DIR.parent.name / PROJECT_DIR.name,
    debug=False,
)
map_figure_saver = figure_saver(**map_figure_saver_kwargs)
for fig_saver in (figure_saver, map_figure_saver):
    fig_saver.experiment = PROJECT_DIR.name

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


def get_model(X_train=None, y_train=None):
    return common_get_model(cache_dir=CACHE_DIR, X_train=X_train, y_train=y_train)


model_score_cache = SimpleCache("model_scores", cache_dir=CACHE_DIR)


@model_score_cache
def get_model_scores(rf=None, X_test=None, X_train=None, y_test=None, y_train=None):
    return common_get_model_scores(rf, X_test, X_train, y_test, y_train)
