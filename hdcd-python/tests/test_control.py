from hdcd import Control, hdcd
import pytest
from hdcd import hdcd
import numpy as np

@pytest.mark.parametrize("segmentation_type, method, kwargs, expected", [
    # minimal_relative_segment_length
    ("bs", "knn", {"minimal_relative_segment_length": 0.05}, [50, 100]),
    ("bs", "knn", {"minimal_relative_segment_length": 0.4}, [60]),
    # minimal_gain_to_split
    ("bs", "change_in_mean", {"minimal_gain_to_split": 0.1}, [50, 100]),
    ("bs", "change_in_mean", {"minimal_gain_to_split": 1}, [50]),
    ("bs", "change_in_mean", {"minimal_gain_to_split": 10}, []),
    # model_selection_alpha
    ("bs", "knn", {"model_selection_alpha": 0.001}, []),
    ("bs", "knn", {"model_selection_alpha": 0.05}, [50, 100]),
])
def test_control_model_selection_parameters(iris_dataset, method, segmentation_type, kwargs, expected):
    result = hdcd(iris_dataset, method, segmentation_type, Control(**kwargs))
    np.testing.assert_array_equal(result.split_points(), expected)


@pytest.mark.parametrize("segmentation_type, kwargs, expected_number_of_segments", [
    # seeded_segments_alpha
    ("sbs", {"minimal_relative_segment_length": 0.05, "seeded_segments_alpha": 1 / np.sqrt(2)}, 50),
    ("sbs", {"minimal_relative_segment_length": 0.05, "seeded_segments_alpha": 0.5}, 25),
    ("bs", {}, 0),
    # number_of_wild_segments
    ("wbs", {"number_of_wild_segments": 10}, 10),
    ("wbs", {"number_of_wild_segments": 25}, 25),
])
def test_control_segmentation_parameters(iris_dataset, segmentation_type, kwargs, expected_number_of_segments):
    result = hdcd(iris_dataset, "change_in_mean", segmentation_type, Control(**kwargs))
    assert(len(result.segments) == expected_number_of_segments)
