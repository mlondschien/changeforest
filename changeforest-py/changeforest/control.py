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
        self.minimal_relative_segment_length = _to_float(
            minimal_relative_segment_length
        )
        self.minimal_gain_to_split = _to_float(minimal_gain_to_split)
        self.model_selection_alpha = _to_float(model_selection_alpha)
        self.number_of_wild_segments = _to_int(number_of_wild_segments)
        self.seeded_segments_alpha = _to_float(seeded_segments_alpha)
        self.seed = _to_int(seed)
        self.random_forest_ntrees = _to_int(random_forest_ntrees)


def _to_float(value):
    if value is None:
        return None
    else:
        return float(value)


def _to_int(value):
    if value is None:
        return None
    else:
        return int(value)
