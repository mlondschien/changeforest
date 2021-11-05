expect_lists_equal = function(l1, l2) {
    expect(
        length(l1) == length(l2),
        paste0("length mismatch. len(l1)=", length(l1), ", len(l2)=", length(l2))
    )

    if (length(l1) == length(l2) & length(l1) > 0) {  # unbelievable that we have to do this in R.
        for (idx in 1 : length(l1)) {
            expect(
                l1[idx] == l2[idx],
                paste0("value mismatch at idx=", idx, ", l1[idx]=", l1[idx], ", l2[idx]=", l2[idx])
            )
        }
    }
}

get_X = function() {
    matrix(
        c(
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
            0, 0, 0, 1, 1, 1, 1, 1, 1, 0
        ),
        ncol=3
    )
}

get_iris = function() {
    library(datasets)
    data(iris)
    as.matrix(iris[, 1:4])
}