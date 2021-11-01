test_that("hdcd_iris", {
    library(datasets)
    data(iris)
    X <- as.matrix(iris[, 1:4])

    expect(hdcd(X, "knn", "bs")$split_points() == c(50, 100))
    expect(hdcd(X, "change_in_mean", "bs")$split_points() == c(50, 100))
    expect(hdcd(X, "random_forest", "bs")$split_points() == c(50, 100))

    expect(hdcd(X, "knn", "sbs")$split_points() == c(50, 100))
    expect(hdcd(X, "change_in_mean", "sbs")$split_points() == c(50, 100))
    expect(hdcd(X, "random_forest", "sbs")$split_points() == c(50, 100))

    expect(hdcd(X, "knn", "wbs")$split_points() == c(50, 100))
    expect(hdcd(X, "change_in_mean", "wbs")$split_points() == c(50, 100))
    expect(hdcd(X, "random_forest", "wbs")$split_points() == c(50, 100))
})
