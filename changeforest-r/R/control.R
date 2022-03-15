#' Storage container for hyperparameters.
#'
#' See rust documentation of changeforest::control::Control for more details.
#' @export
Control = R6::R6Class(
    "control",
    list(
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
        random_forest_n_jobs = "default",

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
