#[derive(Copy, Clone)]
pub struct Control {
    pub minimal_relative_segment_length: f64,
    pub minimal_gain_to_split: f64,
    pub alpha: f64,
    pub number_of_wild_segments: usize,
    pub seeded_segments_alpha: f64,
    pub seed: usize,
}

impl Control {
    pub fn default() -> Control {
        Control {
            minimal_relative_segment_length: 0.1,
            minimal_gain_to_split: 0.,
            alpha: 0.05,
            number_of_wild_segments: 100,
            seeded_segments_alpha: 0.70710678118, // 1 / sqrt(2)
            seed: 0,
        }
    }

    pub fn with_number_of_wild_segments(&mut self, number_of_wild_segments: usize) -> &mut Self {
        self.number_of_wild_segments = number_of_wild_segments;
        self
    }

    pub fn with_seeded_segments_alpha(&mut self, seeded_segments_alpha: f64) -> &mut Self {
        self.seeded_segments_alpha = seeded_segments_alpha;
        self
    }

    pub fn with_minimal_relative_segment_length(
        &mut self,
        minimal_relative_segment_length: f64,
    ) -> &mut Self {
        self.minimal_relative_segment_length = minimal_relative_segment_length;
        self
    }
}
