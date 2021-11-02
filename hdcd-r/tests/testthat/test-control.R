test_that("control", {
    # TODO: Test each control parameter separately.
    X = matrix(
        c(
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 0, 1, 1, 1, 1, 1, 1, 0
        ),
        ncol=3
    )

    expect(hdcd(X, "knn", "bs", Control$new())$split_points() == c(3, 5, 8))
    expect(hdcd(X, "knn", "bs", Control$new(minimal_relative_segment_length=0.4))$split_points() == c(5))
})
    