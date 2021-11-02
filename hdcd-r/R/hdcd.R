BinarySegmentationResult = R6::R6Class(
    "binary_segmentation_result",
    list(
        start = NULL,
        stop = NULL,
        best_split = NULL,
        max_gain = NULL,
        is_significant = NULL,
        gain = NULL,
        left = NULL,
        right = NULL,

        initialize = function(
            start = NULL,
            stop = NULL,
            best_split = NULL,
            max_gain = NULL,
            is_significant = NULL,
            gain = NULL,
            left = NULL,
            right = NULL
        ) {
            self$start = start
            self$stop = stop
            self$best_split = best_split
            self$max_gain = max_gain
            self$is_significant = is_significant
            self$gain = gain
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

            if (!is.null(self$best_split) & self$is_significant) {
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
#' @export
hdcd = function(X,  method, segmentation) {
    result = hdcd_api(X, method, segmentation)

    to_binary_segmentation_result(result)
}

to_data_frame = function(node, parent_indent="", prefix="") {
    name = paste0(prefix, "(", node$start, ", ", node$stop, "]")

    left = data.frame()
    right = data.frame()

    non_last_prefix = paste0(parent_indent, " ", "\u00A6", "--")  # " ¦--"
    last_prefix = paste0(parent_indent, " ", "\u00B0", "--")  # " °--"

    non_last_indent = paste0(parent_indent, " ", "\u00A6", "   ")
    last_indent = paste0(parent_indent, "    ")

    if (!is.null(node$left)) {
        if (is.null(node$right)) {  # left is only and thus last child
            left = to_data_frame(node$left, last_indent, last_prefix)
        } else {  # left is not last child
            left = to_data_frame(node$left, non_last_indent, non_last_prefix)
        }
    }

    if (!is.null(node$right)) {
        right = to_data_frame(node$right, parent_indent=last_indent, prefix=last_prefix)
    }

    frame = rbind(
        data.frame(
            name=name,
            best_split=node$best_split,
            max_gain=node$max_gain,
            is_significant=node$is_significant
        ),
        left,
        right
    )

    # Strings get printed right-aligned. Add spaces to the right to make then seem
    # left-aligned.
    indent = max(sapply(frame$name, nchar))
    frame$name = apply(
        frame,
        1,
        function(row) paste0(row[1], strrep(" ", indent - nchar(row[1])))
    )

    frame
}

print.binary_segmentation_result = function(binary_segmentation_result) {
    print(to_data_frame(binary_segmentation_result))
}