class Control:
    """
    Storage container for hyperparameters.

    See rust documentation of changeforest::control::Control for more details.
    """

    def __init__(
        self,
        minimal_relative_segment_length="default",
        minimal_gain_to_split="default",
        model_selection_alpha="default",
        model_selection_n_permutations="default",
        number_of_wild_segments="default",
        seeded_segments_alpha="default",
        seed="default",
        random_forest_n_estimators="default",
        random_forest_max_depth="default",
        random_forest_max_features="default",
        random_forest_n_jobs="default",
    ):
        self.minimal_relative_segment_length = _to_float(
            minimal_relative_segment_length
        )
        self.minimal_gain_to_split = _to_float(minimal_gain_to_split)
        self.model_selection_alpha = _to_float(model_selection_alpha)
        self.model_selection_n_permutations = _to_int(model_selection_n_permutations)
        self.number_of_wild_segments = _to_int(number_of_wild_segments)
        self.seeded_segments_alpha = _to_float(seeded_segments_alpha)
        self.seed = _to_int(seed)
        self.random_forest_n_estimators = _to_int(random_forest_n_estimators)
        self.random_forest_max_depth = _to_int(random_forest_max_depth)
        self.random_forest_max_features = _to_int(random_forest_max_features)
        self.random_forest_n_jobs = _to_int(random_forest_n_jobs)


def _to_float(value):
    if value is None:
        return None
    elif isinstance(value, str):
        return value
    else:
        return float(value)


def _to_int(value):
    if value is None:
        return None
    elif isinstance(value, str):
        return value
    else:
        return int(value)
