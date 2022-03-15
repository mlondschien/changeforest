test_that("changeforest_iris", {
    X = get_iris()

    expect_lists_equal(changeforest(X, "knn", "bs", Control$new(minimal_relative_segment_length=0.1))$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "change_in_mean", "bs", Control$new(minimal_relative_segment_length=0.1))$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "random_forest", "bs", Control$new(minimal_relative_segment_length=0.1))$split_points(), c(50, 100))

    expect_lists_equal(changeforest(X, "knn", "sbs", Control$new(minimal_relative_segment_length=0.1))$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "change_in_mean", "sbs", Control$new(minimal_relative_segment_length=0.1))$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "random_forest", "sbs", Control$new(minimal_relative_segment_length=0.1))$split_points(), c(50, 100))

    expect_lists_equal(changeforest(X, "knn", "wbs", Control$new(minimal_relative_segment_length=0.1))$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "change_in_mean", "wbs", Control$new(minimal_relative_segment_length=0.1))$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "random_forest", "wbs", Control$new(minimal_relative_segment_length=0.1))$split_points(), c(50, 100))
})
