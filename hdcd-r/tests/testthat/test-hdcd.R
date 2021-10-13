test_that("hdcd_iris", {
    library(datasets)
    data(iris)
    X <- as.matrix(iris[, 1:4])

    expect_equal(hdcd(X, "knn", "bs"), c(50, 100))
    expect_equal(hdcd(X, "change_in_mean", "bs"), c(50, 100))

    expect_equal(hdcd(X, "knn", "sbs"), c(50, 100))
    expect_equal(hdcd(X, "change_in_mean", "sbs"), c(50, 100))
        
    expect_equal(hdcd(X, "knn", "wbs"), c(50, 100))
    expect_equal(hdcd(X, "change_in_mean", "wbs"), c(50, 100))
})