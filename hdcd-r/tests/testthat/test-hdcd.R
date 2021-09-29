test_that("hdcd_iris", {
    library(datasets)
    data(iris)
    X <- as.matrix(iris[, 1:4])
    expect_equal(hdcd(X), c(50, 100))
})