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

#' Plot an object of class `binary_segmentation_result`.
#'
#' @param x An object of class `binary_segmentation_result`.
#' @param ... Not used.
#' @importFrom("graphics", "abline", "lines", "par")
#' @export
plot.binary_segmentation_result = function(x, ...) {
    nodes = c(x)
    gains = list()
    splits = list()
    found_changepoints = list()
    guesses = list()
    n = x$stop

    for (depth in 1:5) {
        if (length(nodes) == 0) {
            break
        }

        new_nodes = list()
        gains = append(gains, list(list()))
        splits = append(splits, list(list()))
        found_changepoints = append(found_changepoints, list(list()))
        guesses = append(guesses, list(list()))

        for (node in nodes){
            splits[[length(splits)]] = append(splits[length(splits)], list(node$start, node$stop))

            if (!is.null(node$optimizer_result) && node$start < node$stop){
                results = node$optimizer_result$gain_results
                result = results[[length(results)]]
                gains[[length(gains)]] = append(gains[[length(gains)]], list(rep(NA, n)))
                # This is the R equivalent of gains[-1][-1]
                gains[[length(gains)]][[length(gains[[length(gains)]])]][(node$start + 1) : (node$stop - 1)] = result$gain[2:length(result$gain)]
                guesses[[length(guesses)]] = append(guesses[[length(guesses)]], c(result$guess))
            } 

            if (node$model_selection_result$is_significant) {
                found_changepoints[[length(found_changepoints)]] = append(found_changepoints[[length(found_changepoints)]], list(node$best_split))
            }

            if (!is.null(node$left)) {
                new_nodes = append(new_nodes, list(node$left))
            }
            if (!is.null(node$right)) {
                new_nodes = append(new_nodes, list(node$right))
            }
        }
        nodes = new_nodes
    }

    depth = length(gains)

    par(mfrow=c(depth, 1))
    for (idx in 1:depth) {
        min_value = Inf
        max_value = -Inf
        for (gain in gains[[idx]]) {
            min_value = min(min_value, min(gain, na.rm=TRUE))
            max_value = max(max_value, max(gain, na.rm=TRUE))
        }

        plot(NULL, type="n", xlim=c(0, n), ylim=c(min_value, max_value), xlab="t", ylab="gain")

        for (jdx in 1:length(gains[[idx]])) {
            lines(gains[[idx]][[jdx]], col="black", lwd=1)
            if (length(guesses[[idx]]) >= jdx) {
                abline(v=guesses[[idx]][[jdx]], col="blue", lwd=2, lty=2)
            }
            abline(v=splits[[idx]][[jdx]], col="black", lwd=2)
            if (length(found_changepoints[[idx]]) >= jdx) {
                abline(v=found_changepoints[[idx]][[jdx]], col="red", lwd=2, lty=3)
            }
        }
    }
}

#' Plot an object of class `optimizer_result`.
#'
#' @param x An object of class `optimizer_result`.
#' @param ... Not used.
#' @importFrom("graphics", "abline", "lines", "par")
#' @export
plot.optimizer_result = function(x, ...) {
    range = (x$start + 1) : (x$stop - 1)

    if (length(x$gain_results) == 1) {
        gain = x$gain_results[[1]]$gain
        plot(range, gain[2 : length(gain)], type="l", xlab="t", ylab="gain")
        abline(v=x$best_split, col="red", lty=1, lwd=2)
    } else {

        par(mfrow=c(4, 2), mai=c(0.6, 0.6, 0.2, 0.2))
        min_gain = Inf
        max_gain = -Inf
        for (idx in 1:4) {
            min_gain = min(min_gain, min(x$gain_results[[idx]]$gain, na.rm=TRUE))
            max_gain = max(max_gain, max(x$gain_results[[idx]]$gain, na.rm=TRUE))
        }

        for (idx in 1:4) {
            gain = x$gain_results[[idx]]$gain
            plot(range, gain[2 : length(gain)], type="l", xlab="split", ylab="gain", ylim=c(min_gain, max_gain))
            abline(v=which.max(x$gain_results[[idx]]$gain) - 1 + x$start, col="red", lty=3, lwd=2)
            abline(v=x$gain_results[[idx]]$guess, col="blue", lty=2, lwd=2)
            
            
            plot((x$start + 1) : x$stop, x$gain_results[[idx]]$predictions, xlab="t", ylab="proba. predictions", ylim=c(0, 1))
            abline(v=x$gain_results[[idx]]$guess, col="blue", lty=2, lwd=2)
        }
    }
}