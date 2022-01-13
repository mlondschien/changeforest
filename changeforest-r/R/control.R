#' Storage container for hyperparameters.
#'
#' See rust documentation of changeforest::control::Control for more details.
#' @export
Control = R6::R6Class(
    "control",
    list(
        minimal_relative_segment_length = NULL,
        minimal_gain_to_split = NULL,
        model_selection_alpha = NULL,
        model_selection_n_permutations = NULL,
        number_of_wild_segments = NULL,
        seeded_segments_alpha = NULL,
        seed = NULL,
        random_forest_n_trees = NULL,
        random_forest_mtry = NULL,     
        random_forest_max_depth = NULL,
        random_forest_n_jobs = NULL,

        initialize = function(
            minimal_relative_segment_length = NULL,
            minimal_gain_to_split = NULL,
            model_selection_alpha = NULL,
            model_selection_n_permutations = NULL,
            number_of_wild_segments = NULL,
            seeded_segments_alpha = NULL,
            seed = NULL,
            random_forest_n_trees = NULL,
            random_forest_mtry = NULL,
            random_forest_max_depth = NULL,
            random_forest_n_jobs = NULL
        ) {
            self$minimal_relative_segment_length = minimal_relative_segment_length
            self$minimal_gain_to_split = minimal_gain_to_split
            self$model_selection_alpha = model_selection_alpha
            self$model_selection_n_permutations = model_selection_n_permutations
            self$number_of_wild_segments = number_of_wild_segments
            self$seeded_segments_alpha = seeded_segments_alpha
            self$seed = seed
            self$random_forest_n_trees = random_forest_n_trees 
            self$random_forest_mtry = random_forest_mtry
            self$random_forest_max_depth = random_forest_max_depth
            self$random_forest_n_jobs = random_forest_n_jobs
        }
    )
)
