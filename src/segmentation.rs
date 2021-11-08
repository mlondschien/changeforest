use crate::optimizer::OptimizerResult;
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
                // See Definition 1 of https://arxiv.org/pdf/2002.06633.pdf
                let n_layers = ((2. * optimizer.control().minimal_relative_segment_length).ln()
                    / optimizer.control().seeded_segments_alpha.ln())
                .ceil();
                let mut segment_length: f64;
                let mut alpha_k: f64;
                let mut n_segments: f64;
                let mut segment_step: f64;
                let mut start: usize;
                let mut stop: usize;
                for k in 1..(n_layers as i32) {
                    alpha_k = optimizer.control().seeded_segments_alpha.powi(k);
                    segment_length = (optimizer.n() as f64) * alpha_k;
                    n_segments = 2. * (1. / alpha_k).ceil() - 1.;
                    segment_step = (optimizer.n() as f64 - segment_length) / (n_segments - 1.);
                    for segment_id in 0..(n_segments as usize) {
                        start = (segment_id as f64 * segment_step) as usize;
                        stop = start + segment_length.ceil() as usize;
                        segments.push(optimizer.find_best_split(start, stop).unwrap());
                    }
                }
            }
            SegmentationType::WBS => {
                let mut rng = StdRng::seed_from_u64(optimizer.control().seed as u64);
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
                // swap_remove does not maintain order of items but is O(1)
                Ok(self.segments.swap_remove(idx_opt))
            }
        }
    }

    pub fn is_significant(&self, optimizer_result: &OptimizerResult) -> bool {
        self.optimizer.is_significant(optimizer_result)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::{testing, Control};
    use rstest::*;

    #[rstest]
    #[case(SegmentationType::BS, vec![])]
    #[case(SegmentationType::SBS, vec![
        (0, 71, 17, 710.0),
        (14, 85, 31, 1704.0),
        (29, 100, 46, 2769.0),
        (0, 51, 12, 510.0),
        (24, 75, 36, 1734.0),
        (49, 100, 61, 3009.0)
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
    #[case(SegmentationType::SBS, (61, 3009.))]
    #[case(SegmentationType::WBS, (60, 2900.))]
    fn test_optimizer(#[case] segmentation_type: SegmentationType, #[case] expected: (usize, f64)) {
        let control = Control::default();
        let optimizer = testing::TrivialOptimizer { control: &control };
        let mut segmentation = Segmentation::new(segmentation_type, &optimizer);

        let result = segmentation.find_best_split(0, 100).unwrap();
        assert_eq!((result.best_split, result.max_gain), expected);
    }
}
