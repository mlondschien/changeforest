test_that("hdcd_iris", {
    library(datasets)
    data(iris)
    X <- as.matrix(iris[, 1:4])

    expect(hdcd(X, "knn", "bs")$best_split %in% c(50, 100))
    expect(hdcd(X, "change_in_mean", "bs")$best_split %in% c(50, 100))
    expect(hdcd(X, "random_forest", "bs")$best_split %in% c(50, 100))

    expect(hdcd(X, "knn", "sbs")$best_split %in% c(50, 100))
    expect(hdcd(X, "change_in_mean", "sbs")$best_split %in% c(50, 100))
    expect(hdcd(X, "random_forest", "sbs")$best_split %in% c(50, 100))
        
    expect(hdcd(X, "knn", "wbs")$best_split %in% c(50, 100))
    expect(hdcd(X, "change_in_mean", "wbs")$best_split %in% c(50, 100))
    expect(hdcd(X, "random_forest", "wbs")$best_split %in% c(50, 100))
})