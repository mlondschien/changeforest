test_that("display", {
    X = get_X()

    result = changeforest(X, "knn", "bs")
    output = capture.output(print(result))

    expected = c(
        "             name best_split max_gain p_value is_significant",
        "1 (0, 10]                  5 8.095522    0.01           TRUE",
        "2  ¦--(0, 5]               3 3.459535    0.01           TRUE",
        "3  ¦    ¦--(0, 3]          1 0.000000    1.00          FALSE",
        "4  ¦    °--(3, 5]         NA       NA      NA          FALSE",
        "5  °--(5, 10]              7 1.383814    1.00          FALSE"
    )

    expect_lists_equal(output, expected)
})
