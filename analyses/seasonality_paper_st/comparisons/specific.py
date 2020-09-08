# -*- coding: utf-8 -*-
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

figure_saver = PaperFigureSaver(
    directories=Path("~") / "tmp" / PROJECT_DIR.parent.name / PROJECT_DIR.name,
    debug=True,
)
map_figure_saver = figure_saver(**map_figure_saver_kwargs)
for fig_saver in (figure_saver, map_figure_saver):
    fig_saver.experiment = PROJECT_DIR.name

memory = get_memory("__".join((PROJECT_DIR.parent.name, PROJECT_DIR.name)), verbose=100)
CACHE_DIR = Path(DATA_DIR) / ".pickle" / PROJECT_DIR.parent.name / PROJECT_DIR.name


def single_ax_multi_ale_1d(
    ax,
    feature_data,
    feature,
    bins=20,
    xlabel=None,
    ylabel=None,
    title=None,
    n_jobs=8,
    verbose=False,
):
    quantile_list = []
    ale_list = []

    for experiment, single_experiment_data in zip(
        tqdm(
            feature_data["experiment"],
            desc="Calculating feature ALEs",
            disable=not verbose,
        ),
        feature_data["single_experiment_data"],
    ):
        cache = SimpleCache(
            f"{experiment}_{feature}_ale_{bins}",
            cache_dir=CACHE_DIR / "ale",
            verbose=10 if verbose else 0,
        )
        try:
            quantiles, ale = cache.load()
        except NoCachedDataError:
            model = single_experiment_data["model"]
            model.n_jobs = n_jobs

            X_train = single_experiment_data["X_train"]

            with parallel_backend("threading", n_jobs=n_jobs):
                quantiles, ale = first_order_ale_quant(
                    model.predict, X_train, feature, bins=bins
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

    for plot_kwargs, quantiles, ale in zip(
        feature_data["plot_kwargs"], quantile_list, ale_list
    ):
        # Interpolate each of the quantiles relative to the accumulated final quantiles.
        ax.plot(
            np.interp(quantiles, final_quantiles, mod_quantiles),
            ale,
            **{"marker": "o", "ms": 3, **plot_kwargs},
        )

        ax.set_xticks(mod_quantiles[::2])
        ax.set_xticklabels(
            [
                t if t != "0.0e+0" else "0"
                for t in _sci_format(final_quantiles[::2], scilim=0)
            ]
        )
        ax.xaxis.set_tick_params(rotation=18)

        ax.grid(alpha=0.4, linestyle="--")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title)


def multi_model_ale_1d(
    feature_name,
    experiment_data,
    experiment_plot_kwargs,
    lags=(0, 1, 3, 6, 9),
    bins=20,
    title=None,
    n_jobs=8,
    verbose=False,
    figure_saver=None,
    single_figsize=(5.4, 1.5),
    legend_bbox=(0.5, 0.5),
):
    assert set(experiment_data) == set(experiment_plot_kwargs)
    plotted_experiments = set()

    # Compile data for later plotting.
    comp_data = {}

    for lag in tqdm(lags, desc="Lags", disable=not verbose):
        if lag:
            feature = f"{feature_name} {-lag} Month"
        else:
            feature = feature_name

        feature_data = defaultdict(list)

        experiment_count = 0
        for experiment, single_experiment_data in experiment_data.items():

            # Skip experiments that do not contain this feature.
            if feature not in single_experiment_data["X_train"]:
                continue

            experiment_count += 1
            plotted_experiments.add(experiment)

            # Data required to calculate the ALEs.
            feature_data["experiment"].append(experiment)
            feature_data["single_experiment_data"].append(single_experiment_data)
            feature_data["plot_kwargs"].append(experiment_plot_kwargs[experiment])

        if experiment_count <= 1:
            # We need at least two models for a comparison.
            continue

        comp_data[feature] = feature_data

    n_plots = len(comp_data)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=np.array(single_figsize) * np.array([n_cols, n_rows])
    )

    # Disable unused axes.
    if len(axes.flatten()) > n_plots:
        for ax in axes.flatten()[-(len(axes.flatten()) - n_plots) :]:
            ax.axis("off")

    for ax, feature, feature_data in zip(axes.flatten(), comp_data, comp_data.values()):
        single_ax_multi_ale_1d(
            ax,
            feature_data=feature_data,
            feature=feature,
            bins=bins,
            xlabel=shorten_features(feature).replace(fill_name(""), ""),
            n_jobs=n_jobs,
            verbose=verbose,
        )

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        t = np.format_float_scientific(x, precision=1, unique=False, exp_digits=1)
        return t if t != "0.0e+0" else "0"

    for ax in axes.flatten()[:n_plots]:
        ax.yaxis.set_major_formatter(major_formatter)

    for row_axes in axes:
        row_axes[0].set_ylabel("ALE")

    fig.tight_layout()

    lines = []
    labels = []
    for experiment in sort_experiments(plotted_experiments):
        lines.append(Line2D([0], [0], **experiment_plot_kwargs[experiment]))
        labels.append(experiment_plot_kwargs[experiment]["label"])

    fig.legend(
        lines,
        labels,
        loc="center",
        bbox_to_anchor=legend_bbox,
        ncol=len(labels) if len(labels) <= 6 else 6,
    )

    if figure_saver is not None:
        figure_saver.save_figure(
            fig,
            f"{shorten_features(feature_name).replace(' ', '_').lower()}_ale_comp",
            sub_directory="ale_comp",
        )
