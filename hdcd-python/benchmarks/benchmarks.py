# Inspired by https://github.com/scikit-learn/scikit-learn/blob/main/asv_benchmarks/\
# benchmarks/common.py
import timeit
from abc import ABC, abstractmethod
from typing import Any
from hdcd import hdcd

from .data import load_letters, load_iris


class Benchmark:
    """Class for benchmarking."""

    # Measure wall time instead of CPU usage
    timer = timeit.default_timer

    param_names = ["data", "method", "segmentation"]
    params = (
        ["iris"],
        ["random_forest", "knn", "change_in_mean"],
        ["bs", "sbs", "wbs"],
    )

    def setup(self, *params):
        """Make data for benchmarking."""
        self._data = {}
        self._data["letters"] = load_letters().sort_values("class").drop(columns="class").to_numpy()
        self._data["iris"] = load_iris().sort_values("class").drop(columns="class").to_numpy()

    def time_hdcd(self, *params):
        data, method, segmentation = params
        X = self._data[data]
        result = hdcd(X, method, segmentation)
