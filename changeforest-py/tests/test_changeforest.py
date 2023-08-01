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


def test_changeforest_repr_segments(iris_dataset):
    result = changeforest(
        iris_dataset,
        "random_forest",
        "bs",
        control=Control(forbidden_segments=[(0, 49), (101, 120)]),
    )
    assert (
        result.__repr__()
        == """\
                    best_split max_gain p_value
(0, 150]                    50     95.1   0.005
 ¦--(0, 50]                                    
 °--(50, 150]              100   52.799   0.005
     ¦--(50, 100]           53    6.892   0.315
     °--(100, 150]         136   -3.516    0.68\
"""  # noqa: W291
    )


def test_changeforest_repr_segments2(iris_dataset):
    result = changeforest(
        iris_dataset,
        "random_forest",
        "bs",
        control=Control(forbidden_segments=[(49, 101)]),
    )
    assert (
        result.__repr__()
        == """\
                    best_split max_gain p_value
(0, 150]                    49   87.462   0.005
 ¦--(0, 49]                  2   -8.889   0.995
 °--(49, 150]              102   41.237   0.005
     ¦--(49, 102]                              
     °--(102, 150]         136    1.114    0.36\
"""  # noqa: W291
    )
