import numpy as np

from .changeforest import OptimizerResult

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False


def plot_two_step_search_result(optimizer_result):
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


def plot_gain_result(gain_result):
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


def plot_optimizer_result(optimizer_result):
    if len(optimizer_result.gain_results) == 4:
        return plot_two_step_search_result(optimizer_result)
    else:
        return plot_gain_result(optimizer_result.gain_results[-1])


OptimizerResult.plot = plot_optimizer_result
