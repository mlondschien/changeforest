test_that("display", {
    X = matrix(
        c(
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 0, 1, 1, 1, 1, 1, 1, 0
        ),
        ncol=3
    )

    result = changeforest(X, "knn", "bs")
    output = capture.output(print(result))

    expected = c(
        "                  name best_split  max_gain p_value is_significant",
        "1  (0, 10]                      5  5.379459    0.03           TRUE",
        "2   ¦--(0, 5]                   2  1.383814    0.01           TRUE",
        "3   ¦    ¦--(0, 2]             NA        NA      NA          FALSE",
        "4   ¦    °--(2, 5]              3  0.000000    0.01           TRUE",
        "5   ¦        ¦--(2, 3]         NA        NA      NA          FALSE",
        "6   ¦        °--(3, 5]         NA        NA      NA          FALSE",
        "7   °--(5, 10]                  7 -4.616186    0.01           TRUE",
        "8       ¦--(5, 7]              NA        NA      NA          FALSE",
        "9       °--(7, 10]              8  0.000000    0.01           TRUE",
        "10          ¦--(7, 8]          NA        NA      NA          FALSE",
        "11          °--(8, 10]         NA        NA      NA          FALSE"
    )

    expect_lists_equal(output, expected)
})
