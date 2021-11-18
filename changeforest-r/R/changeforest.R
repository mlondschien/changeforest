BinarySegmentationResult = R6::R6Class(
    "binary_segmentation_result",
    list(
        start = NULL,
        stop = NULL,
        best_split = NULL,
        max_gain = NULL,
        model_selection_result = NULL,
        p_value = NULL,
        is_significant = NULL,
        optimizer_result = NULL,
        segments = NULL,
        left = NULL,
        right = NULL,

        initialize = function(
            start = NULL,
            stop = NULL,
            best_split = NULL,
            max_gain = NULL,
            model_selection_result = NULL,
            p_value = NULL,
            is_significant = NULL,
            optimizer_result = NULL,
            segments = NULL,
            left = NULL,
            right = NULL
        ) {
            self$start = start
            self$stop = stop
            self$best_split = best_split
            self$max_gain = max_gain
            self$model_selection_result = model_selection_result
            self$optimizer_result = optimizer_result
            self$p_value = p_value
            self$is_significant = is_significant
            self$segments = segments
            self$left = left
            self$right = right
        },

        depth = function() {
            max(
                ifelse(is.null(self$left), 0, 1 + self$left$depth()),
                ifelse(is.null(self$right), 0, 1 + self$right$depth())
            )
        },

        split_points = function() {
            split_points = c()

            if (! is.null(self$left)) {
                split_points = append(split_points, self$left$split_points())
            }

            if (!is.null(self$best_split) & self$model_selection_result$is_significant) {
                split_points = append(split_points, c(self$best_split))
            }

            if (! is.null(self$right)) {
                split_points = append(split_points, self$right$split_points())
            }

            split_points
        }
    )
)

to_binary_segmentation_result = function(result) {
    if (!is.null(result$left)) result$left = to_binary_segmentation_result(result$left)
    if (!is.null(result$right)) result$right = to_binary_segmentation_result(result$right)

    result = do.call(BinarySegmentationResult$new, result)
    class(result) = "binary_segmentation_result"

    result
}

#' Find change points in a time series.
#'
#' @param X Numerical matrix with time series.
#' @param method Either 'knn','change_in_mean' of 'random_forest'.
#' @param segmentation Either 'bs', 'sbs' or 'wbs'.
#' @param control Object of class Control containing hyperparameters.
#' @export
changeforest = function(X,  method, segmentation, control=Control$new()) {
    result = changeforest_api(X, method, segmentation, control)
    to_binary_segmentation_result(result)
}
