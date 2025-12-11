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
(0, 150]                    50   95.867   0.005
 ¦--(0, 50]                  2   -14.37       1
 °--(50, 150]              100   52.853   0.005
     ¦--(50, 100]           53    5.152    0.27
     °--(100, 150]         102   -6.981     0.9\
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
(0, 150]                    50   94.844   0.005
 ¦--(0, 50]                                    
 °--(50, 150]              100   52.853   0.005
     ¦--(50, 100]           53    5.152    0.28
     °--(100, 150]         147  -12.303   0.975\
"""  # noqa W291
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
(0, 150]                    49   87.437   0.005
 ¦--(0, 49]                  2   -13.76    0.96
 °--(49, 150]              102   34.621   0.005
     ¦--(49, 102]                              
     °--(102, 150]         138    0.301   0.475\
"""  # noqa W291
    )
