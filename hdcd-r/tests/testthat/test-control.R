# Check that all hyperparameters in control are correctly passed to rust.
test_that("control", {
    X = get_X()
    X_iris = get_iris()

    # minimal_relative_segment_length
    expect_lists_equal(hdcd(X, "knn", "bs", Control$new(minimal_relative_segment_length=0.05))$split_points(), c(2, 3, 5,7, 8))
    expect_lists_equal(hdcd(X, "knn", "bs", Control$new(minimal_relative_segment_length=0.4))$split_points(), c(5))

    # minimal_gain_to_split
    expect_lists_equal(hdcd(X, "change_in_mean", "bs", Control$new(minimal_gain_to_split=0.1))$split_points(), c(3, 5, 8))
    expect_lists_equal(hdcd(X, "change_in_mean", "bs", Control$new(minimal_gain_to_split=0.2))$split_points(), c(5))
    expect_lists_equal(hdcd(X, "change_in_mean", "bs", Control$new(minimal_gain_to_split=1))$split_points(), c())

    # model_selection_alpha
    expect_lists_equal(hdcd(X_iris, "random_forest", "bs", Control$new(model_selection_alpha=0.001))$split_points(),  c())
    expect_lists_equal(hdcd(X_iris, "random_forest", "bs", Control$new(model_selection_alpha=0.05))$split_points(), c(50, 100))

    # seeded_segments_alpha
    expect_lists_equal(hdcd(X, "change_in_mean", "bs")$segments, c())
    expect_equal(length(hdcd(X_iris, "change_in_mean", "sbs", Control$new(minimal_relative_segment_length=0.05, seeded_segments_alpha=1/sqrt(2)))$segments), 50)
    expect_equal(length(hdcd(X_iris, "change_in_mean", "sbs", Control$new(minimal_relative_segment_length=0.05, seeded_segments_alpha=1/2))$segments), 25)
    expect_equal(length(hdcd(X, "change_in_mean", "sbs", Control$new(minimal_relative_segment_length=0.05, seeded_segments_alpha=1/2))$segments), 10)

    # number_of_wild_segments
    expect_equal(length(hdcd(X_iris, "change_in_mean", "wbs", Control$new(number_of_wild_segments=10))$segments), 10)
    expect_equal(length(hdcd(X_iris, "change_in_mean", "wbs", Control$new(number_of_wild_segments=5))$segments), 5)
})

