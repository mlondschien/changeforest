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

    expect(hdcd(X, "knn", "bs", Control$new(minimal_relative_segment_length=0.05))$split_points() == c(2, 3, 5,7, 8))
    expect(hdcd(X, "knn", "bs", Control$new(minimal_relative_segment_length=0.4))$split_points() == c(5))

    # TODO: Fix change_in_mean.is_significant.
    # expect(hdcd(X, "knn", "bs", Control$new(minimal_gain_to_split=0.5))$split_points() == c(2, 3, 5,7, 8))
})
    