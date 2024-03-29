# Check that all hyperparameters in control are correctly passed to rust.
# TODO: Figure out how to make testing in R less of a pain.
test_that("control", {
    X = get_X()
    X_iris = get_iris()

    # minimal_relative_segment_length
    expect_lists_equal(changeforest(X, "knn", "bs", Control$new(minimal_relative_segment_length=0.05))$split_points(), c(3, 5))
    expect_lists_equal(changeforest(X, "knn", "bs", Control$new(minimal_relative_segment_length=0.4))$split_points(), c(5))

    # minimal_gain_to_split
    expect_lists_equal(changeforest(X, "change_in_mean", "bs", Control$new(minimal_gain_to_split=0.1))$split_points(), c(3, 5))
    expect_lists_equal(changeforest(X, "change_in_mean", "bs", Control$new(minimal_gain_to_split=0.2))$split_points(), c(3, 5))
    expect_lists_equal(changeforest(X, "change_in_mean", "bs", Control$new(minimal_gain_to_split=10))$split_points(), c())

    # model_selection_alpha
    expect_lists_equal(changeforest(X_iris, "random_forest", "bs", Control$new(model_selection_alpha=0.001))$split_points(),  c())
    expect_lists_equal(changeforest(X_iris, "random_forest", "bs", Control$new(model_selection_alpha=0.05))$split_points(), c(50, 100))

    # seeded_segments_alpha
    expect_equal(length(changeforest(X, "change_in_mean", "bs")$segments), 1)
    expect_equal(length(changeforest(X_iris, "change_in_mean", "sbs", Control$new(minimal_relative_segment_length=0.05, seeded_segments_alpha=1/sqrt(2)))$segments), 44 + 5)
    expect_equal(length(changeforest(X_iris, "change_in_mean", "sbs", Control$new(minimal_relative_segment_length=0.05, seeded_segments_alpha=1/2))$segments), 25 + 5)
    expect_equal(length(changeforest(X, "change_in_mean", "sbs", Control$new(minimal_relative_segment_length=0.05, seeded_segments_alpha=1/2))$segments), 10 + 1)

    # number_of_wild_segments
    expect_equal(length(changeforest(X_iris, "change_in_mean", "wbs", Control$new(number_of_wild_segments=10))$segments), 10 + 5)
    expect_equal(length(changeforest(X_iris, "change_in_mean", "wbs", Control$new(number_of_wild_segments=5))$segments), 5 + 5)

    # seed
    result = changeforest(X_iris, "random_forest", "wbs", Control$new(number_of_wild_segments=10, seed=42))
    expect_equal(result$segments[[1]]$start, 5)
    result = changeforest(X_iris, "random_forest", "wbs", Control$new(number_of_wild_segments=10, seed=12))
    expect_equal(result$segments[[1]]$start, 72)

    # random_forest_n_estimators
    expect_lists_equal(changeforest(X_iris, "random_forest", "bs", Control$new(random_forest_n_estimators=1))$split_points(), c(47, 99))
    expect_lists_equal(changeforest(X_iris, "random_forest", "bs", Control$new(random_forest_n_estimators=100))$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(random_forest_n_estimators=1))$split_points(), c())
    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(random_forest_n_estimators=10))$split_points(), c(3, 5))

    # max_features
    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(random_forest_n_estimators=2, random_forest_max_depth=1, random_forest_max_features=1))$split_points(), c())
    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(random_forest_n_estimators=2, random_forest_max_depth=1, random_forest_max_features=0.4))$split_points(), c())
    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(random_forest_n_estimators=2, random_forest_max_depth=1, random_forest_max_features="sqrt"))$split_points(), c())
    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(random_forest_n_estimators=2, random_forest_max_depth=1))$split_points(), c())  # sqrt is the default value.
    


    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(random_forest_n_estimators=2, random_forest_max_depth=1, random_forest_max_features=2))$split_points(), c(3, 5, 8))
    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(random_forest_n_estimators=2, random_forest_max_depth=1, random_forest_max_features=0.7))$split_points(), c(3, 5, 8))

    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(random_forest_n_estimators=2, random_forest_max_depth=1, random_forest_max_features=NULL))$split_points(), c(3, 5, 8))
    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(random_forest_n_estimators=2, random_forest_max_depth=1, random_forest_max_features=3))$split_points(), c(3, 5, 8))

    # model_selection_n_permutations
    expect_lists_equal(changeforest(X_iris, "random_forest", "bs", Control$new(model_selection_n_permutations=10))$split_points(), c())
})

