from pathlib import Path

import numpy as np
import pytest

_IRIS_FILE = "iris.csv"
# maturin build maps the parent directory of the python package to
# "local_dependencies/changeforest". This allows tests to run e.g. on conda-forge.
local_dependencies = Path(__file__).resolve().parents[1] / "local_dependencies"
if local_dependencies.exists():
    _IRIS_PATH = local_dependencies / "changeforest" / "testdata" / _IRIS_FILE
else:
    _IRIS_PATH = Path(__file__).resolve().parents[2] / "testdata" / _IRIS_FILE


@pytest.fixture(scope="module")
def iris_dataset():
    return np.loadtxt(_IRIS_PATH, skiprows=1, delimiter=",", usecols=(0, 1, 2, 3))


@pytest.fixture(scope="module")
def X_test():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )


@pytest.fixture(scope="module")
def X_correlated():
    X = np.zeros((100, 2))

    rng = np.random.default_rng(7)

    X[:, 0] = rng.normal(0, 1, 100)
    X[0:50, 1] = rng.normal(0, 1, 50)
    X[50:100, 1] = X[50:100, 0]

    return X
