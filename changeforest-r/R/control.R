Control = R6::R6Class(
    "control",
    list(
        minimal_relative_segment_length = NULL,
        minimal_gain_to_split = NULL,
        model_selection_alpha = NULL,
        number_of_wild_segments = NULL,
        seeded_segments_alpha = NULL,
        seed = NULL,
        random_forest_ntrees = NULL,       

        initialize = function(
            minimal_relative_segment_length = NULL,
            minimal_gain_to_split = NULL,
            model_selection_alpha = NULL,
            number_of_wild_segments = NULL,
            seeded_segments_alpha = NULL,
            seed = NULL,
            random_forest_ntrees = NULL 
        ) {
            self$minimal_relative_segment_length = minimal_relative_segment_length
            self$minimal_gain_to_split = minimal_gain_to_split
            self$model_selection_alpha = model_selection_alpha
            self$number_of_wild_segments = number_of_wild_segments
            self$seeded_segments_alpha = seeded_segments_alpha
            self$seed = seed
            self$random_forest_ntrees = random_forest_ntrees 
        }
    )
)
