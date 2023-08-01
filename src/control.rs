use biosphere::{MaxFeatures, RandomForestParameters};

/// Storage container for hyperparameters.
#[derive(Clone)]
pub struct Control {
    /// Segments with length smaller than `2 * n * minimal_relative_segment_length` will
    /// not be split.
    pub minimal_relative_segment_length: f64,
    /// Only keep split point if the gain exceeds `minimal_gain_to_split`. Relevant for
    /// change in mean. Note that this is relative to the number of observations.
    /// Use value motivated by BIC `minimal_gain_to_split = log(n_samples) * n_features / n_samples`
    /// if `None`.
    pub minimal_gain_to_split: Option<f64>,
    /// Type two error in model selection to be approximated. Relevant for classifier-
    /// based changepoint detection.
    pub model_selection_alpha: f64,
    /// Number of permutations for model selection in classifier-based change point
    /// detection.
    pub model_selection_n_permutations: usize,
    /// Number of randomly drawn segments. Corresponds to parameter `M` in
    /// https://arxiv.org/pdf/1411.0858.pdf.
    pub number_of_wild_segments: usize,
    /// Decay parameter in seeded binary segmentation. Should be in `[1/2, 1)`, with a
    /// value close to 1 resulting in many segments. Corresponds to `\alpha` in
    /// https://arxiv.org/pdf/2002.06633.pdf.
    pub seeded_segments_alpha: f64,
    /// Seed used for segmentation.
    pub seed: u64,
    /// Hyperparameters for random forests.
    pub random_forest_parameters: RandomForestParameters,
    /// Segments of indexes were no segmentation is allowed.
    pub forbidden_segments: Option<Vec<(usize, usize)>>,
}

impl Control {
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Control {
        Control {
            minimal_relative_segment_length: 0.01,
            minimal_gain_to_split: None,
            model_selection_alpha: 0.02,
            model_selection_n_permutations: 199,
            number_of_wild_segments: 100,
            seeded_segments_alpha: std::f64::consts::FRAC_1_SQRT_2, // 1 / sqrt(2)
            seed: 0,
            random_forest_parameters: RandomForestParameters::default()
                .with_max_depth(Some(8))
                .with_max_features(MaxFeatures::Sqrt)
                .with_n_jobs(Some(-1)),
            forbidden_segments: None,
        }
    }

    pub fn with_minimal_relative_segment_length(
        mut self,
        minimal_relative_segment_length: f64,
    ) -> Self {
        if (minimal_relative_segment_length >= 0.5) | (minimal_relative_segment_length <= 0.) {
            panic!(
                "minimal_relative_segment_length needs to be strictly between 0 and 0.5 Got {minimal_relative_segment_length}"
            );
        }
        self.minimal_relative_segment_length = minimal_relative_segment_length;
        self
    }

    pub fn with_minimal_gain_to_split(mut self, minimal_gain_to_split: Option<f64>) -> Self {
        self.minimal_gain_to_split = minimal_gain_to_split;
        self
    }

    pub fn with_model_selection_alpha(mut self, model_selection_alpha: f64) -> Self {
        if (model_selection_alpha >= 1.) | (model_selection_alpha <= 0.) {
            panic!(
                "model_selection_alpha needs to be strictly between 0 and 1. Got {model_selection_alpha}"                
            );
        }
        self.model_selection_alpha = model_selection_alpha;
        self
    }

    pub fn with_model_selection_n_permutations(
        mut self,
        model_selection_n_permutations: usize,
    ) -> Self {
        self.model_selection_n_permutations = model_selection_n_permutations;
        self
    }

    pub fn with_number_of_wild_segments(mut self, number_of_wild_segments: usize) -> Self {
        self.number_of_wild_segments = number_of_wild_segments;
        self
    }

    pub fn with_seeded_segments_alpha(mut self, seeded_segments_alpha: f64) -> Self {
        if (1. <= seeded_segments_alpha) | (seeded_segments_alpha <= 0.) {
            panic!(
                "seeded_segments_alpha needs to be strictly between 0 and 1. Got {seeded_segments_alpha}"                
            );
        }
        self.seeded_segments_alpha = seeded_segments_alpha;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_random_forest_parameters(
        mut self,
        random_forest_parameters: RandomForestParameters,
    ) -> Self {
        self.random_forest_parameters = random_forest_parameters;
        self
    }

    pub fn with_forbidden_segments(
        mut self,
        forbidden_segments: Option<Vec<(usize, usize)>>,
    ) -> Self {
        // check that segments are well specified
        if let Some(ref _forbidden_segments) = forbidden_segments {
            for el in _forbidden_segments.iter() {
                if el.0 > el.1 {
                    panic!("Forbidden segments must be specified as [(a,b), ...] where a <= b!");
                }
            }
        }
        self.forbidden_segments = forbidden_segments;
        self
    }
}
