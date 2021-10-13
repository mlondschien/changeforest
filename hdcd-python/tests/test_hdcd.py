from pathlib import Path

import numpy as np
import pytest
from hdcdpython import hdcd

_IRIS_FILE = "iris.csv"
_IRIS_PATH = Path(__file__).resolve().parents[2] / "testdata" / _IRIS_FILE


@pytest.fixture()
def iris_dataset():
    return np.loadtxt(_IRIS_PATH, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3))

@pytest.mark.parametrize("method", ["knn", "change_in_mean"])
@pytest.mark.parametrize("segmentation_type", ["sbs", "wbs", "bs"])
def test_hdcd(iris_dataset, method, segmentation_type):
    result = hdcd(iris_dataset, method, segmentation_type)
    np.testing.assert_array_equal(result, [50, 100])
