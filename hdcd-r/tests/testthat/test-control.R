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
})

