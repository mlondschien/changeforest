import numpy as np
import pytest

from changeforest import Control, changeforest


@pytest.mark.parametrize("method", ["knn", "change_in_mean", "random_forest"])
@pytest.mark.parametrize("segmentation_type", ["sbs", "wbs", "bs"])
def test_changeforest(iris_dataset, method, segmentation_type):
    result = changeforest(
        iris_dataset,
        method,
        segmentation_type,
        control=Control(minimal_relative_segment_length=0.1),
    )
    np.testing.assert_array_equal(result.split_points(), [50, 100])


def test_changeforest_repr(iris_dataset):
    result = changeforest(iris_dataset, "random_forest", "bs")
    assert (
        result.__repr__()
        == """\
                    best_split max_gain p_value
(0, 150]                    50   96.233   0.005
 ¦--(0, 50]                  2  -14.191       1
 °--(50, 150]              100   52.799   0.005
     ¦--(50, 100]           53     5.44   0.245
     °--(100, 150]         136   -2.398   0.875\
"""
    )
