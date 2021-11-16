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
            p_value=node$p_value,
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

#' Print an object of class `binary_segmentation_result`.
#'
#' @param x An object of class `binary_segmentation_result`.
#' @param ... Not used.
#' @export
print.binary_segmentation_result = function(x, ...) {
    print(to_data_frame(x))
}