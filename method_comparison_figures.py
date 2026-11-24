"""Compare the number of distinct models calibrated by each search.

Regenerate the figure with::

    PYTHONHASHSEED=0 python method_comparison_figures.py

The hash seed fixes the order in which the searches encounter candidate
models of the same size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from modelSelector.criterion import AIC, AIC_c, BIC
from modelSelector.model import LinearRegressionModel
from modelSelector.selection_aglorithm.efficient_exhaustive_search import (
    efficient_exhaustive_search,
)
from modelSelector.selection_aglorithm.jumping_backwards_exhaustive_search import (
    jumping_backwards_exhaustive_search,
)


CRITERIA = (AIC, AIC_c, BIC)
CRITERION_TITLES = ("AIC", r"AIC$_c$", "BIC")
SALIENCIES = ("coefficient", "hessian")
METHOD_STYLES = {
    "exhaustive": ("Efficient exhaustive", "o", -0.19),
    "coefficient": (r"Jumping backwards ($|\delta_i|$)", "^", 0.0),
    "hessian": (r"Jumping backwards ($H_{ii}|\delta_i|$)", "D", 0.19),
}


def make_data(
    n: int, n_parameters: int, snr: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Create a linear data set with a known support."""
    x = rng.normal(size=(n, 12))
    signal = x[:, :n_parameters].sum(axis=1)
    noise_sd = np.sqrt(np.var(signal) / snr)
    y = signal + rng.normal(scale=noise_sd, size=n)
    return x, y


def jumping_backwards_calibrations(
    x: np.ndarray,
    y: np.ndarray,
    criterion,
    saliency: str = "coefficient",
) -> int:
    """Count the distinct models fitted by jumping backwards search."""
    result, _ = jumping_backwards_exhaustive_search(
        super_model=LinearRegressionModel("M_" + "1" * x.shape[1]),
        criterion=criterion(),
        data_x=x,
        data_y=y,
        saliency=saliency,
    )
    return result.n_fitted_models


def exhaustive_calibrations(x: np.ndarray, y: np.ndarray, criterion) -> int:
    """Count the distinct models fitted by efficient exhaustive search."""
    result = efficient_exhaustive_search(
        super_model=LinearRegressionModel("M_" + "1" * x.shape[1]),
        criterion=criterion(),
        data_x=x,
        data_y=y,
    )
    return result.n_fitted_models


def collect_results(
    parameter_sizes: Iterable[int], datasets: int = 5, snr: float = 10000
) -> list[dict]:
    """Run all three methods on each simulated dataset and criterion."""
    rng = np.random.default_rng(2024)
    results = []
    for size in parameter_sizes:
        for data_set in range(datasets):
            x, y = make_data(100, size, snr, rng)
            for criterion in CRITERIA:
                counts = {"exhaustive": exhaustive_calibrations(x, y, criterion)}
                for saliency in SALIENCIES:
                    counts[saliency] = jumping_backwards_calibrations(
                        x, y, criterion, saliency
                    )
                for method, count in counts.items():
                    results.append(
                        {"size": size, "data": data_set, "criterion": criterion,
                         "method": method, "count": count}
                    )
    return results


def plot_results(
    parameter_sizes: Sequence[int] = tuple(range(2, 12)),
    output: str | Path = "method_comparison_with_hessian.pdf",
) -> None:
    """Save a three-panel comparison of calibration counts as a vector PDF."""
    results = collect_results(parameter_sizes)
    fig, axes = plt.subplots(1, len(CRITERIA), figsize=(16, 5), layout="constrained")
    data_sets = sorted({row["data"] for row in results})
    colormap = plt.get_cmap("tab10")
    colors = {data_set: colormap(index % 10) for index, data_set in enumerate(data_sets)}

    for axis, criterion, title in zip(axes, CRITERIA, CRITERION_TITLES):
        for method, (_, marker, offset) in METHOD_STYLES.items():
            for data_set in data_sets:
                rows = [
                    row for row in results
                    if row["criterion"] == criterion
                    and row["method"] == method
                    and row["data"] == data_set
                ]
                axis.scatter(
                    [row["size"] + offset for row in rows],
                    [row["count"] for row in rows],
                    marker=marker,
                    color=colors[data_set],
                    s=28,
                    alpha=0.8,
                )
        axis.set(title=title, xticks=parameter_sizes)
        axis.set_axisbelow(True)
        axis.grid(alpha=0.25)

    method_handles = [
        Line2D([], [], linestyle="none", marker=marker, color="#334155", label=label)
        for label, marker, _ in METHOD_STYLES.values()
    ]
    dataset_handles = [
        Line2D([], [], linestyle="none", marker="s", color=colors[data_set],
               label=f"Dataset {data_set}")
        for data_set in data_sets
    ]

    fig.legend(handles=method_handles + dataset_handles, loc="outside upper center",
               ncol=len(method_handles) + len(dataset_handles), frameon=False)
    fig.supxlabel(r"Number of parameters in the true model ($|S|$)")
    fig.supylabel("Distinct models calibrated")

    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    plot_results()
