import numpy as np

from .changeforest import BinarySegmentationResult, OptimizerResult

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False


def _plot_two_step_search_result(optimizer_result):
    if not MATPLOTLIB_INSTALLED:
        raise ImportError("The matplotlib package is required for OptimizerResult.plot")

    min_gain = min(
        min(gain_result.gain) for gain_result in optimizer_result.gain_results
    )
    max_gain = max(
        max(gain_result.gain) for gain_result in optimizer_result.gain_results
    )
    delta = max_gain - min_gain
    (ymin_gain, ymax_gain) = (min_gain - 0.05 * delta, max_gain + 0.05 * delta)
    gain_range = (min_gain - 0.1 * delta, max_gain + 0.1 * delta)

    ymin_proba = -0.05
    ymax_proba = 1.05
    proba_range = (-0.1, 1.1)

    fig, axes = plt.subplots(nrows=4, ncols=2)

    for idx in range(4):
        axes[idx, 0].plot(
            range(optimizer_result.start, optimizer_result.stop),
            optimizer_result.gain_results[idx].gain,
            color="k",
        )
        axes[idx, 0].set_ylim(*gain_range)

        axes[idx, 0].vlines(
            optimizer_result.gain_results[idx].guess,
            ymin=ymin_gain,
            ymax=ymax_gain,
            linestyles="dashed",
            color="#4477AA",  # blue
            linewidth=2,
        )
        axes[idx, 0].vlines(
            optimizer_result.start
            + np.nanargmax(optimizer_result.gain_results[idx].gain),
            ymin=ymin_gain,
            ymax=ymax_gain,
            linestyles="dotted",
            color="#EE6677",  # red
            linewidth=2,
        )
        axes[idx, 0].set_ylabel("approx. gain")

        axes[idx, 1].set_ylim(*proba_range)
        axes[idx, 1].vlines(
            optimizer_result.gain_results[idx].guess,
            ymin=ymin_proba,
            ymax=ymax_proba,
            linestyles="dashed",
            color="#4477AA",  # blue
            linewidth=2,
        )
        axes[idx, 1].scatter(
            range(optimizer_result.start, optimizer_result.stop),
            optimizer_result.gain_results[idx].predictions,
            s=2,
            c="k",
        )

        axes[idx, 1].set_ylabel("proba. predictions")
        if idx < 3:
            axes[idx, 0].set_xticklabels([])
            axes[idx, 1].set_xticklabels([])

    axes[-1, 0].set_xlabel("split")
    axes[-1, 1].set_xlabel("t")

    fig.align_ylabels(axes[:, 0])
    fig.align_ylabels(axes[:, 1])
    fig.tight_layout()
    return fig


def _plot_gain_result(gain_result):
    if not MATPLOTLIB_INSTALLED:
        raise ImportError("The matplotlib package is required for OptimizerResult.plot")

    fig, axes = plt.subplots()

    axes.plot(range(gain_result.start, gain_result.stop), gain_result.gain, color="k")
    axes.vlines(
        np.nanmax(gain_result.gain) + gain_result.start,
        linestyles="dotted",
        color="#EE6677",  # red
        linewidth=2,
    )

    axes.set_xlabel("split")
    axes.set_ylabel("gain")

    fig.tight_layout()
    return fig


def _plot_optimizer_result(optimizer_result):
    if len(optimizer_result.gain_results) == 4:
        return _plot_two_step_search_result(optimizer_result)
    else:
        return _plot_gain_result(optimizer_result.gain_results[-1])


def _plot_binary_segmentation_result(binary_segmentation_result, max_depth=5):
    if not MATPLOTLIB_INSTALLED:
        raise ImportError(
            "The matplotlib package is required for BinarySegmentationResult.plot"
        )

    nodes = [binary_segmentation_result]
    gains = []
    splits = []
    found_changepoints = []
    guesses = []
    n = binary_segmentation_result.stop

    for _ in range(0, max_depth):
        new_nodes = []

        if len(nodes) == 0:
            break

        gains.append([])
        found_changepoints.append([])
        splits.append([])
        guesses.append([])

        for node in nodes:
            splits[-1].append(node.start)
            splits[-1].append(node.stop)

            if node.optimizer_result is not None:
                result = node.optimizer_result.gain_results[-1]
                gains[-1].append(np.full(n, np.nan))
                gains[-1][-1][node.start : node.stop] = result.gain  # noqa: E203
                guesses[-1].append(result.guess)

            if node.model_selection_result.is_significant:
                found_changepoints[-1].append(node.best_split)

            if node.left is not None:
                new_nodes.append(node.left)
            if node.right is not None:
                new_nodes.append(node.right)

        nodes = new_nodes

    depth = len(gains)

    fig, axes = plt.subplots(nrows=depth)
    if depth == 1:
        axes = [axes]

    for idx in range(depth):

        for gain in gains[idx]:
            axes[idx].plot(gain, color="black")

        ymin, ymax = axes[idx].get_ylim()
        new_ymin, new_ymax = ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin)

        axes[idx].vlines(splits[idx], color="black", ymin=new_ymin, ymax=new_ymax)

        axes[idx].vlines(
            guesses[idx],
            color="#4477AA",  # blue
            ymin=ymin,
            ymax=ymax,
            linewidth=2,
            linestyles="dashed",
        )

        axes[idx].vlines(
            found_changepoints[idx],
            color="#EE6677",  # red
            ymin=ymin,
            ymax=ymax,
            linewidth=2,
            linestyles="dotted",
        )

        axes[idx].set_xlim(0, n)
        axes[idx].set_ylim(new_ymin, new_ymax)
        axes[idx].set_ylabel("approx. gain")
        if idx < depth - 1:
            axes[idx].set_xticklabels([])

    axes[-1].set_xlabel("split")

    return fig


OptimizerResult.plot = _plot_optimizer_result
BinarySegmentationResult.plot = _plot_binary_segmentation_result
