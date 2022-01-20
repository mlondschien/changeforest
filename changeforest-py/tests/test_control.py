import numpy as np
import pytest

from changeforest import Control, changeforest


@pytest.mark.parametrize(
    "data, segmentation_type, method, kwargs, expected",
    [
        # minimal_relative_segment_length
        ("iris", "bs", "knn", {"minimal_relative_segment_length": 0.05}, [50, 100]),
        ("iris", "bs", "knn", {"minimal_relative_segment_length": 0.4}, [60]),
        # minimal_gain_to_split
        ("iris", "bs", "change_in_mean", {"minimal_gain_to_split": 0.1}, [50, 100]),
        ("iris", "bs", "change_in_mean", {"minimal_gain_to_split": 1}, [50]),
        ("iris", "bs", "change_in_mean", {"minimal_gain_to_split": 10}, []),
        ("iris", "bs", "change_in_mean", {"minimal_gain_to_split": 10.0}, []),
        # model_selection_alpha
        ("iris", "bs", "knn", {"model_selection_alpha": 0.001}, []),
        ("iris", "bs", "knn", {"model_selection_alpha": 0.05}, [50, 100]),
        # random_forest_ntree
        # This is impressive and unexpected.
        ("iris", "bs", "random_forest", {"random_forest_n_trees": 1}, [37, 52, 99]),
        ("iris", "bs", "random_forest", {"random_forest_n_trees": 100}, [50, 100]),
        # Use X_test instead
        ("X_test", "bs", "random_forest", {"random_forest_n_trees": 1}, []),
        ("X_test", "bs", "random_forest", {"random_forest_n_trees": 1.0}, []),
        ("X_test", "bs", "random_forest", {"random_forest_n_trees": 100}, [5]),
        ("X_correlated", "bs", "random_forest", {"random_forest_max_depth": 1}, []),
        ("X_correlated", "bs", "random_forest", {"random_forest_max_depth": 2}, [49]),
        ("iris", "bs", "random_forest", {"model_selection_n_permutations": 10}, []),
    ],
)
def test_control_model_selection_parameters(
    iris_dataset,
    X_test,
    X_correlated,
    data,
    method,
    segmentation_type,
    kwargs,
    expected,
):
    if data == "iris":
        X = iris_dataset
    elif data == "X_test":
        X = X_test
    else:
        X = X_correlated

    result = changeforest(X, method, segmentation_type, Control(**kwargs))
    np.testing.assert_array_equal(result.split_points(), expected)


@pytest.mark.parametrize(
    "segmentation_type, kwargs, expected_number_of_segments",
    [
        # seeded_segments_alpha
        (
            "sbs",
            {
                "minimal_relative_segment_length": 0.05,
                "seeded_segments_alpha": 1 / np.sqrt(2),
            },
            44,
        ),
        (
            "sbs",
            {"minimal_relative_segment_length": 0.05, "seeded_segments_alpha": 0.5},
            25,
        ),
        ("bs", {}, 0),
        # number_of_wild_segments
        ("wbs", {"number_of_wild_segments": 10}, 10),
        ("wbs", {"number_of_wild_segments": 25}, 25),
    ],
)
def test_control_segmentation_parameters(
    iris_dataset, segmentation_type, kwargs, expected_number_of_segments
):
    result = changeforest(
        iris_dataset, "change_in_mean", segmentation_type, Control(**kwargs)
    )
    # For each split, add evaluation on left / right segment to segments.
    expected_number_of_segments = (
        expected_number_of_segments + 2 * len(result.split_points()) + 1
    )
    assert len(result.segments) == expected_number_of_segments


def test_control_seed(iris_dataset):
    result = changeforest(
        iris_dataset,
        "random_forest",
        "wbs",
        Control(seed=42, number_of_wild_segments=10),
    )
    assert result.segments[0].start == 5
    assert abs(result.segments[0].max_gain - 11.38617) < 1e-5

    result = changeforest(
        iris_dataset,
        "random_forest",
        "wbs",
        Control(seed=12, number_of_wild_segments=10),
    )
    assert result.segments[0].start == 21
    assert abs(result.segments[0].max_gain - 44.98708) < 1e-5
