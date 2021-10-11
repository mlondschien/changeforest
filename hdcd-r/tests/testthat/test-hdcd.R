test_that("hdcd_iris", {
    library(datasets)
    data(iris)
    X <- as.matrix(iris[, 1:4])
    expect_equal(hdcd(X, "knn"), c(50, 100))
    expect_equal(hdcd(X, "change_in_mean"), c(50, 100))
})