# -*- coding: utf-8 -*-
import importlib.util
import sys
import warnings
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if sys.path[0] != str(PROJECT_DIR.parent):
    sys.path.insert(0, str(PROJECT_DIR.parent))

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.utils.deprecation"
)
from common import *
from common import _sci_format

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


def load_experiment_data(folders):
    """Load data from specified experiments.

    Args:
        folders (iterable of {str, Path}): Folder names corresponding to the
            experiments to load data for.

    Returns:
        dict of dict: Keys are the given `folders` and the loaded data types.

    """
    data = defaultdict(dict)

    for experiment in folders:
        # Load the different experiments' modules.
        spec = importlib.util.spec_from_file_location(
            f"{experiment}_specific",
            str(PROJECT_DIR.parent / experiment / "specific.py"),
        )
        data[experiment]["module"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data[experiment]["module"])
        data[experiment].update(
            {
                key: data
                for key, data in zip(
                    (
                        "endog_data",
                        "exog_data",
                        "master_mask",
                        "filled_datasets",
                        "masked_datasets",
                        "land_mask",
                    ),
                    data[experiment]["module"].get_offset_data(),
                )
            }
        )
        data[experiment]["model"] = data[experiment]["module"].get_model()
        data[experiment].update(
            {
                key: data
                for key, data in zip(
                    ("X_train", "X_test", "y_train", "y_test"),
                    data[experiment]["module"].data_split_cache.load(),
                )
            }
        )

    return data


def multi_model_ale_plot_1d(
    model_X_cols_kwargs,
    fig_name,
    xlabel=None,
    ylabel=None,
    title=None,
    n_jobs=8,
    verbose=False,
    figure_saver=None,
    figsize=(8, 2),
):
    fig, ax = plt.subplots(figsize=figsize)  # Make sure to plot on a new figure.
    quantile_list = []
    ale_list = []
    for experiment, data_dict, feature, _ in tqdm(
        model_X_cols_kwargs, desc="Calculating feature ALEs", disable=not verbose
    ):
        bins = 20
        cache = SimpleCache(
            f"{experiment}_{feature}_ale_{bins}",
            cache_dir=CACHE_DIR / "ale",
            verbose=10 if verbose else 0,
        )
        try:
            quantiles, ale = cache.load()
        except NoCachedDataError:
            with parallel_backend("threading", n_jobs=n_jobs):
                model = data_dict[experiment]["model"]
                model.n_jobs = n_jobs
                quantiles, ale = first_order_ale_quant(
                    model.predict, data_dict[experiment]["X_train"], feature, bins=bins
                )
                cache.save((quantiles, ale))

        quantile_list.append(quantiles)
        ale_list.append(ale)

    # Construct quantiles from the individual quantiles, minimising the amount of interpolation.
    combined_quantiles = np.vstack([quantiles[None] for quantiles in quantile_list])

    final_quantiles = np.mean(combined_quantiles, axis=0)
    # Account for extrema.
    final_quantiles[0] = np.min(combined_quantiles)
    final_quantiles[-1] = np.max(combined_quantiles)

    mod_quantiles = np.arange(len(quantiles))
    for (_, _, _, plot_kwargs), quantiles, ale in zip(
        model_X_cols_kwargs, quantile_list, ale_list
    ):
        # Interpolate each of the quantiles relative to the accumulated final quantiles.
        ax.plot(
            np.interp(quantiles, final_quantiles, mod_quantiles),
            ale,
            **{"marker": "o", "ms": 3, **plot_kwargs},
        )

    ax.legend(loc="best")
    ax.set_xticks(mod_quantiles)
    ax.set_xticklabels(_sci_format(final_quantiles, scilim=0.6))
    ax.xaxis.set_tick_params(rotation=45)
    ax.grid(alpha=0.4, linestyle="--")

    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if figure_saver is not None:
        figure_saver.save_figure(fig, fig_name, sub_directory="multi_ale")
