test_that("hdcd_iris", {
    X = get_iris()

    expect_lists_equal(hdcd(X, "knn", "bs")$split_points(), c(50, 100))
    expect_lists_equal(hdcd(X, "change_in_mean", "bs")$split_points(), c(50, 100))
    expect_lists_equal(hdcd(X, "random_forest", "bs")$split_points(), c(50, 100))

    expect_lists_equal(hdcd(X, "knn", "sbs")$split_points(), c(50, 100))
    expect_lists_equal(hdcd(X, "change_in_mean", "sbs")$split_points(), c(50, 100))
    expect_lists_equal(hdcd(X, "random_forest", "sbs")$split_points(), c(50, 100))

    expect_lists_equal(hdcd(X, "knn", "wbs")$split_points(), c(50, 100))
    expect_lists_equal(hdcd(X, "change_in_mean", "wbs")$split_points(), c(50, 100))
    expect_lists_equal(hdcd(X, "random_forest", "wbs")$split_points(), c(50, 100))
})
