#' Storage container for hyperparameters.
#'
#' See rust documentation of changeforest::control::Control,
#' biosphere::forest::RandomForestParameters and biosphere::tree::DecisionTreeParameters
#' for more details.
#'
#' @param minimal_relative_segment_length Segments with length smaller than
#' \code{2 * n * minimal_relative_segment_length} will not be split. Equal to 0.01 by default.
#' @param minimal_gain_to_split Only keep split point if the gain exceeds
#' \code{minimal_gain_to_split}. Relevant for change in mean. Use value motivated by BIC
#' \code{minimal_gain_to_split = log(n_samples) * n_features / n_samples} by default.
#' @param model_selection_alpha Type two error in model selection to be approximated.
#' Relevant for classifier-based changepoint detection. Equal to 0.02 by default.
#' @param model_selection_n_permutations Number of permutations for model selection in
#' classifier-based change point detection. Equal to 199 by default.
#' @param number_of_wild_segments Number of randomly drawn segments. Corresponds to
#' parameter \code{M} in https://arxiv.org/pdf/1411.0858.pdf. Only relevant if
#' \code{segmentation='wbs'}. Equal to 100 by default.
#' @param seeded_segments_alpha Decay parameter in seeded binary segmentation. Should
#' be in \code{[1/2, 1)}, with a value close to 1 resulting in many segments. Corresponds to
#' \eqn{\alpha} in https://arxiv.org/pdf/2002.06633.pdf. Only relevant if
#' \code{segmentatin='sbs'}. Equal to \eqn{1 / \sqrt{2}} by default.
#' @param seed Seed for segmentation and random forest. Only relevant for
#' \code{segmentation='wbs'} or \code{method='random_forest'}.
#' @param random_forest_n_estimators Parameter passed to random forest classifier if
#' \code{method='random_forest'}. Equal to 100 by default.
#' @param random_forest_max_features Parameter passed to random forest classifier if
#' \code{method='random_forest'}. Equal to \eqn{\sqrt{d}} by default.
#' @param random_forest_max_depth Parameter passed to random forest classifier if
#' \code{method='random_forest'}. Equal to 8 by default.
#' @param random_forest_n_jobs Parameter passed to random forest classifier if
#' \code{method='random_forest'}. Use all cores if -1. Equal to -1 by default.
#'
#' @return Object of class Control containing hyperparameters.
#' @export
Control = R6::R6Class(
    "control",
    list(
        #' @field minimal_relative_segment_length Segments with length smaller than 
        #' \code{2 * n * minimal_relative_segment_length} will not be split. Equal to
        #' 0.01 by default.
        minimal_relative_segment_length = "default",
        #' @field minimal_gain_to_split Only keep split point if the gain exceeds
        #' \code{minimal_gain_to_split}. Relevant for change in mean. Use value motivated by BIC
        #' \code{minimal_gain_to_split = log(n_samples) * n_features / n_samples} by default.
        minimal_gain_to_split = "default",
        #' @field model_selection_alpha Type two error in model selection to be approximated.
        #' Relevant for classifier-based changepoint detection. Equal to 0.02 by default.
        model_selection_alpha = "default",
        #' @field model_selection_n_permutations Number of permutations for model selection in
        #' classifier-based change point detection. Equal to 199 by default.
        model_selection_n_permutations = "default",
        #' @field number_of_wild_segments Number of randomly drawn segments. Corresponds to
        #' parameter \code{M} in https://arxiv.org/pdf/1411.0858.pdf. Only relevant if
        #' \code{segmentation='wbs'}. Equal to 100 by default.
        number_of_wild_segments = "default",
        #' @field seeded_segments_alpha Decay parameter in seeded binary segmentation. Should
        #' be in \code{[1/2, 1)}, with a value close to 1 resulting in many segments. Corresponds to
        #' \eqn{\alpha} in https://arxiv.org/pdf/2002.06633.pdf. Only relevant if
        #' \code{segmentatin='sbs'}. Equal to \eqn{1 / \sqrt{2}} by default.
        seeded_segments_alpha = "default",
        #' @field seed Seed for segmentation and random forest. Only relevant for
        #' \code{segmentation='wbs'} or \code{method='random_forest'}.
        seed = "default",
        #' @field random_forest_n_estimators Parameter passed to random forest classifier if
        #' \code{method='random_forest'}. Equal to 100 by default.
        random_forest_n_estimators = "default",
        #' @field random_forest_max_features Parameter passed to random forest classifier if
        #' \code{method='random_forest'}. Equal to \eqn{\sqrt{d}} by default.
        random_forest_max_features = "default",    
        #' @field random_forest_max_depth Parameter passed to random forest classifier if
        #' \code{method='random_forest'}. Equal to 8 by default. 
        random_forest_max_depth = "default",
        #' @field random_forest_n_jobs Parameter passed to random forest classifier if
        #' \code{method='random_forest'}. Use all cores if -1. Equal to -1 by default.
        random_forest_n_jobs = "default",

        #' @description
        #' Create a new object of class \code{binary_segmentation_resutl}.
        #' @param minimal_relative_segment_length Segments with length smaller than
        #' \code{2 * n * minimal_relative_segment_length} will not be split. Equal to 0.01 by default.
        #' @param minimal_gain_to_split Only keep split point if the gain exceeds
        #' \code{minimal_gain_to_split}. Relevant for change in mean. Use value motivated by BIC
        #' \code{minimal_gain_to_split = log(n_samples) * n_features / n_samples} by default.
        #' @param model_selection_alpha Type two error in model selection to be approximated.
        #' Relevant for classifier-based changepoint detection. Equal to 0.02 by default.
        #' @param model_selection_n_permutations Number of permutations for model selection in
        #' classifier-based change point detection. Equal to 199 by default.
        #' @param number_of_wild_segments Number of randomly drawn segments. Corresponds to
        #' parameter \code{M} in https://arxiv.org/pdf/1411.0858.pdf. Only relevant if
        #' \code{segmentation='wbs'}. Equal to 100 by default.
        #' @param seeded_segments_alpha Decay parameter in seeded binary segmentation. Should
        #' be in \code{[1/2, 1)}, with a value close to 1 resulting in many segments. Corresponds to
        #' \eqn{\alpha} in https://arxiv.org/pdf/2002.06633.pdf. Only relevant if
        #' \code{segmentatin='sbs'}. Equal to \eqn{1 / \sqrt{2}} by default.
        #' @param seed Seed for segmentation and random forest. Only relevant for
        #' \code{segmentation='wbs'} or \code{method='random_forest'}.
        #' @param random_forest_n_estimators Parameter passed to random forest classifier if
        #' \code{method='random_forest'}. Equal to 100 by default.
        #' @param random_forest_max_features Parameter passed to random forest classifier if
        #' \code{method='random_forest'}. Equal to \eqn{\sqrt{d}} by default.
        #' @param random_forest_max_depth Parameter passed to random forest classifier if
        #' \code{method='random_forest'}. Equal to 8 by default.
        #' @param random_forest_n_jobs Parameter passed to random forest classifier if
        #' \code{method='random_forest'}. Use all cores if -1. Equal to -1 by default.
        #' @return A new object of class \code{binary_segmentation_resutl}.
        initialize = function(
            minimal_relative_segment_length = "default",
            minimal_gain_to_split = "default",
            model_selection_alpha = "default",
            model_selection_n_permutations = "default",
            number_of_wild_segments = "default",
            seeded_segments_alpha = "default",
            seed = "default",
            random_forest_n_estimators = "default",
            random_forest_max_features = "default",
            random_forest_max_depth = "default",
            random_forest_n_jobs = "default"
        ) {
            self$minimal_relative_segment_length = minimal_relative_segment_length
            self$minimal_gain_to_split = minimal_gain_to_split
            self$model_selection_alpha = model_selection_alpha
            self$model_selection_n_permutations = model_selection_n_permutations
            self$number_of_wild_segments = number_of_wild_segments
            self$seeded_segments_alpha = seeded_segments_alpha
            self$seed = seed
            self$random_forest_n_estimators = random_forest_n_estimators 
            self$random_forest_max_features = random_forest_max_features
            self$random_forest_max_depth = random_forest_max_depth
            self$random_forest_n_jobs = random_forest_n_jobs
        }
    )
)
