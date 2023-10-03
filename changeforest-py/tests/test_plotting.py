import pytest

from changeforest import Control, changeforest


@pytest.mark.parametrize("method", ["knn", "change_in_mean", "random_forest"])
@pytest.mark.parametrize("segmentation_type", ["bs", "sbs", "wbs"])
def test_plot_binary_segmentation_result(iris_dataset, method, segmentation_type):
    result = changeforest(
        iris_dataset,
        method,
        segmentation_type,
        control=Control(minimal_relative_segment_length=0.1),
    )
    result.plot().show()


@pytest.mark.parametrize("method", ["knn", "change_in_mean", "random_forest"])
@pytest.mark.parametrize("segmentation_type", ["bs", "sbs", "wbs"])
def test_plot_optimizer_result(iris_dataset, method, segmentation_type):
    result = changeforest(
        iris_dataset,
        method,
        segmentation_type,
        control=Control(minimal_relative_segment_length=0.1),
    )
    result.optimizer_result.plot().show()
