use crate::optimizer::OptimizerResult;
use crate::ModelSelectionResult;
use crate::Optimizer;
use rand::{
    distributions::{Distribution, Uniform},
    rngs::StdRng,
    SeedableRng,
};
pub enum SegmentationType {
    BS,
    WBS,
    SBS,
}

pub struct Segmentation<'a> {
    pub segments: Vec<OptimizerResult>,
    optimizer: &'a dyn Optimizer,
}

impl<'a> Segmentation<'a> {
    pub fn new(segmentation_type: SegmentationType, optimizer: &'a dyn Optimizer) -> Self {
        Segmentation {
            segments: Self::get_segments(optimizer, segmentation_type),
            optimizer,
        }
    }

    fn get_segments(
        optimizer: &dyn Optimizer,
        segmentation_type: SegmentationType,
    ) -> Vec<OptimizerResult> {
        let mut segments = vec![];
        match segmentation_type {
            SegmentationType::BS => (),
            SegmentationType::SBS => {
                let minimal_segment_length = f64::max(
                    2. * (optimizer.control().minimal_relative_segment_length
                        * optimizer.n() as f64)
                        .ceil(),
                    2.,
                );
                // See Definition 1 of https://arxiv.org/pdf/2002.06633.pdf
                let n_layers = ((minimal_segment_length / optimizer.n() as f64).ln()
                    / optimizer.control().seeded_segments_alpha.ln())
                .ceil();
                let mut segment_length: f64;
                let mut alpha_k: f64;
                let mut n_segments: usize;
                let mut segment_step: f64;
                let mut start: usize;
                let mut stop: usize;
                for k in 1..(n_layers as i32) {
                    alpha_k = optimizer.control().seeded_segments_alpha.powi(k); // (1/alpha)^(k-1)
                    segment_length = (optimizer.n() as f64) * alpha_k; // l_k
                    n_segments = 2 * ((1. / alpha_k) as f32).ceil() as usize - 1; // n_k
                    segment_step =
                        (optimizer.n() as f64 - segment_length) / (n_segments - 1) as f64; // s_k
                    for segment_id in 0..n_segments {
                        start = ((segment_id as f64 * segment_step) as f32) as usize;
                        // start + segment_length > n through floating point errors in
                        // n_segments, e.g. for n = 20'000, alpha_k = 1/sqrt(2), k=6
                        stop = (start + (segment_length as f32).ceil() as usize).min(optimizer.n());
                        if let Ok(optimizer_result) = optimizer.find_best_split(start, stop) {
                            segments.push(optimizer_result)
                        }
                    }
                }
            }
            SegmentationType::WBS => {
                let mut rng = StdRng::seed_from_u64(optimizer.control().seed);
                let dist = Uniform::from(0..(optimizer.n() + 1));

                let mut start: usize;
                let mut stop: usize;

                while segments.len() < optimizer.control().number_of_wild_segments {
                    start = dist.sample(&mut rng);
                    stop = dist.sample(&mut rng);
                    if start < stop {
                        if let Ok(optimizer_result) = optimizer.find_best_split(start, stop) {
                            segments.push(optimizer_result)
                        }
                    }
                }
            }
        }
        segments
    }
}

impl<'a> Segmentation<'a> {
    pub fn find_best_split(&mut self, start: usize, stop: usize) -> Result<OptimizerResult, &str> {
        match self.optimizer.find_best_split(start, stop) {
            Err(e) => Err(e),
            Ok(optimizer_result) => {
                let mut idx_opt = self.segments.len();
                let mut best_gain = optimizer_result.max_gain;

                for (idx, current_result) in self
                    .segments
                    .iter()
                    .enumerate()
                    .filter(|(_, res)| (res.start >= start) & (res.stop <= stop))
                {
                    if current_result.max_gain > best_gain {
                        best_gain = current_result.max_gain;
                        idx_opt = idx
                    }
                }

                self.segments.push(optimizer_result);

                Ok(self.segments[idx_opt].clone())
            }
        }
    }

    pub fn model_selection(&self, optimizer_result: &OptimizerResult) -> ModelSelectionResult {
        self.optimizer.model_selection(optimizer_result)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{testing, Control};
    use rstest::*;

    #[rstest]
    #[case(0.05, std::f64::consts::FRAC_1_SQRT_2, vec![
        (0, 71), (14, 85), (29, 100), (0, 50), (25, 75),
        (50, 100), (0, 36), (16, 52), (32, 68), (48, 84),
        (64, 100), (0, 25), (12, 37), (25, 50), (37, 62),
        (50, 75), (62, 87), (75, 100), (0, 18), (8, 26),
        (16, 34), (24, 42), (32, 50), (41, 59), (49, 67),
        (57, 75), (65, 83), (74, 92), (82, 100), (0, 13),
        (6, 19), (12, 25), (18, 31), (25, 38), (31, 44),
        (37, 50), (43, 56), (50, 63), (56, 69), (62, 75),
        (68, 81), (75, 88), (81, 94), (87, 100)
    ])]
    #[case(0.12, std::f64::consts::FRAC_1_SQRT_2, vec![
        (0, 71), (14, 85), (29, 100), (0, 50), (25, 75),
        (50, 100), (0, 36), (16, 52), (32, 68), (48, 84),
        (64, 100), (0, 25), (12, 37), (25, 50), (37, 62),
        (50, 75), (62, 87), (75, 100)
    ])]
    #[case(0.12, 0.5, vec![
        (0, 50), (25, 75), (50, 100),
        (0, 25), (12, 37), (25, 50), (37, 62),
        (50, 75), (62, 87), (75, 100)
    ])]
    fn test_sbs_segments(
        #[case] minimal_relative_segment_length: f64,
        #[case] seeded_segments_alpha: f64,
        #[case] expected: Vec<(usize, usize)>,
    ) {
        let control = Control::default()
            .with_minimal_relative_segment_length(minimal_relative_segment_length)
            .with_seeded_segments_alpha(seeded_segments_alpha);

        let optimizer = testing::TrivialOptimizer { control: &control };
        let segmentation = Segmentation::new(SegmentationType::SBS, &optimizer);

        assert_eq!(&segmentation.segments.len(), &expected.len());

        for ((start, stop), result) in expected.iter().zip(segmentation.segments) {
            assert_eq!(*start, result.start);
            assert_eq!(*stop, result.stop);
        }
    }

    #[rstest]
    #[case(SegmentationType::BS, vec![])]
    #[case(SegmentationType::SBS, vec![
        (0, 71, 17, 710.0),
        (14, 85, 31, 1704.0),
        (29, 100, 46, 2769.0),
        (0, 50, 12, 500.0),
        (25, 75, 37, 1750.0),
        (50, 100, 62, 3000.0)
    ])]
    #[case(SegmentationType::WBS, vec![
        (73, 78, 74, 415.0),
        (2, 59, 16, 684.0),
        (26, 77, 38, 1836.0),
        (22, 80, 36, 1856.0),
        (75, 97, 80, 1870.0)
    ])]
    fn test_generate_segments(
        #[case] segmentation_type: SegmentationType,
        #[case] expected: Vec<(usize, usize, usize, f64)>,
    ) {
        let control = Control::default()
            .with_number_of_wild_segments(5)
            .with_minimal_relative_segment_length(0.2);
        let optimizer = testing::TrivialOptimizer { control: &control };
        let segmentation = Segmentation::new(segmentation_type, &optimizer);

        assert_eq!(&segmentation.segments.len(), &expected.len());

        for ((start, stop, best_split, max_gain), result) in
            expected.iter().zip(segmentation.segments)
        {
            assert_eq!(*start, result.start);
            assert_eq!(*stop, result.stop);
            assert_eq!(*best_split, result.best_split);
            assert_eq!(*max_gain, result.max_gain);
        }
    }

    #[rstest]
    #[case(SegmentationType::BS, (25, 1000.))]
    #[case(SegmentationType::SBS, (62, 3000.))]
    #[case(SegmentationType::WBS, (60, 2900.))]
    fn test_optimizer(#[case] segmentation_type: SegmentationType, #[case] expected: (usize, f64)) {
        let control = Control::default();
        let optimizer = testing::TrivialOptimizer { control: &control };
        let mut segmentation = Segmentation::new(segmentation_type, &optimizer);

        let result = segmentation.find_best_split(0, 100).unwrap();
        assert_eq!((result.best_split, result.max_gain), expected);
    }
}
