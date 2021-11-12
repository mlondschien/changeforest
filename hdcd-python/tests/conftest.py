from pathlib import Path

import numpy as np
import pytest

_IRIS_FILE = "iris.csv"
_IRIS_PATH = Path(__file__).resolve().parents[2] / "testdata" / _IRIS_FILE


@pytest.fixture(scope="module")
def iris_dataset():
    return np.loadtxt(_IRIS_PATH, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3))
