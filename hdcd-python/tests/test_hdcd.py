import pytest
from hdcd import hdcd
import numpy as np

@pytest.mark.parametrize("method", ["knn", "change_in_mean", "random_forest"])
@pytest.mark.parametrize("segmentation_type", ["sbs", "wbs", "bs"])
def test_hdcd(iris_dataset, method, segmentation_type):
    result = hdcd(iris_dataset, method, segmentation_type)
    np.testing.assert_array_equal(result.split_points(), [50, 100])
