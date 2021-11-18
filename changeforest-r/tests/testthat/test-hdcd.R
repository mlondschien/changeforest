test_that("changeforest_iris", {
    X = get_iris()

    expect_lists_equal(changeforest(X, "knn", "bs")$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "change_in_mean", "bs")$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "random_forest", "bs")$split_points(), c(50, 100))

    expect_lists_equal(changeforest(X, "knn", "sbs")$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "change_in_mean", "sbs")$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "random_forest", "sbs")$split_points(), c(50, 100))

    expect_lists_equal(changeforest(X, "knn", "wbs")$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "change_in_mean", "wbs")$split_points(), c(50, 100))
    expect_lists_equal(changeforest(X, "random_forest", "wbs")$split_points(), c(50, 100))
})
