class Control:
    """
    Storage container for hyperparameters.

    See rust documentation of changeforest::control::Control for more details.
    """

    def __init__(
        self,
        minimal_relative_segment_length=None,
        minimal_gain_to_split=None,
        model_selection_alpha=None,
        number_of_wild_segments=None,
        seeded_segments_alpha=None,
        seed=None,
        random_forest_ntrees=None,
    ):
        self.minimal_relative_segment_length = minimal_relative_segment_length
        self.minimal_gain_to_split = minimal_gain_to_split
        self.model_selection_alpha = model_selection_alpha
        self.number_of_wild_segments = number_of_wild_segments
        self.seeded_segments_alpha = seeded_segments_alpha
        self.seed = seed
        self.random_forest_ntrees = random_forest_ntrees
