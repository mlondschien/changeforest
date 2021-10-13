use crate::{Control, Optimizer};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::StdRng,
    SeedableRng,
};
use std::cell::RefCell;

pub enum SegmentationType {
    BS,
    WBS,
    SBS,
}

pub struct Segmentation<'a> {
    segmentation_type: SegmentationType,
    segments: RefCell<Vec<(usize, usize, usize, f64)>>,
    optimizer: &'a dyn Optimizer,
}

impl<'a> Segmentation<'a> {
    pub fn new(segmentation_type: SegmentationType, optimizer: &'a dyn Optimizer) -> Self {
        Segmentation {
            segmentation_type,
            segments: RefCell::new(vec![]),
            optimizer,
        }
    }

    pub fn generate_segments(&mut self, optimizer: &dyn Optimizer) {
        let mut segments = vec![];
        match self.segmentation_type {
            SegmentationType::BS => (),
            SegmentationType::SBS => {
                // See Definition 1 of https://arxiv.org/pdf/2002.06633.pdf
                let n_layers = ((2. * self.control().minimal_relative_segment_length).ln()
                    / self.control().seeded_segments_alpha.ln())
                .ceil();
                let mut segment_length: f64;
                let mut alpha_k: f64;
                let mut n_segments: f64;
                let mut segment_step: f64;
                let mut start: usize;
                let mut stop: usize;
                for k in 1..(n_layers as i32) {
                    alpha_k = self.control().seeded_segments_alpha.powi(k);
                    segment_length = (self.n() as f64) * alpha_k;
                    n_segments = 2. * (1. / alpha_k).ceil() - 1.;
                    segment_step = (self.n() as f64 - segment_length) / (n_segments - 1.);
                    for segment_id in 0..(n_segments as usize) {
                        start = (segment_id as f64 * segment_step) as usize;
                        stop = start + segment_length.ceil() as usize;
                        let (best_split, max_gain) =
                            optimizer.find_best_split(start, stop).unwrap();
                        segments.push((start, stop, best_split, max_gain));
                    }
                }
            }
            SegmentationType::WBS => {
                let mut rng = StdRng::seed_from_u64(self.control().seed as u64);
                let dist = Uniform::from(0..(self.n() + 1));

                let mut start: usize;
                let mut stop: usize;

                while segments.len() < self.control().number_of_wild_segments {
                    start = dist.sample(&mut rng);
                    stop = dist.sample(&mut rng);
                    if start < stop {
                        if let Ok((best_split, max_gain)) = optimizer.find_best_split(start, stop) {
                            segments.push((start, stop, best_split, max_gain))
                        }
                    }
                }
            }
        }
        self.segments.replace(segments);
    }
}

impl<'a> Optimizer for Segmentation<'a> {
    fn find_best_split(&self, start: usize, stop: usize) -> Result<(usize, f64), &str> {
        match self.optimizer.find_best_split(start, stop) {
            Err(e) => return Err(e),
            Ok((segment_best_split, segment_max_gain)) => {
                let mut max_gain: f64 = segment_max_gain;
                let mut best_split: usize = segment_best_split;

                for (_, _, _best_split, _max_gain) in self
                    .segments
                    .borrow()
                    .iter()
                    .filter(|(s1, s2, _, _)| (start <= *s1) & (stop >= *s2))
                {
                    if *_max_gain > max_gain {
                        max_gain = *_max_gain;
                        best_split = *_best_split;
                    }
                }

                self.segments.borrow_mut().push((
                    start,
                    stop,
                    segment_best_split,
                    segment_max_gain,
                ));
                Ok((best_split, max_gain))
            }
        }
    }

    fn control(&self) -> &'a Control {
        self.optimizer.control()
    }

    fn n(&self) -> usize {
        self.optimizer.n()
    }

    fn is_significant(&self, start: usize, stop: usize, split: usize, max_gain: f64) -> bool {
        self.optimizer.is_significant(start, stop, split, max_gain)
    }
}
#[cfg(test)]
mod tests {

    use super::*;
    use crate::testing;
    use rstest::*;

    #[rstest]
    #[case(SegmentationType::BS, vec![])]
    #[case(SegmentationType::SBS, vec![
        (0, 71, 17, 35.5),
        (14, 85, 31, 35.5),
        (29, 100, 46, 35.5),
        (0, 50, 12, 25.0),
        (12, 62, 24, 25.0),
        (25, 75, 37, 25.0),
        (37, 87, 49, 25.0),
        (50, 100, 62, 25.0)
    ])]
    #[case(SegmentationType::WBS, vec![
        (73, 78, 74, 2.5),
        (2, 59, 16, 28.5),
        (26, 77, 38, 25.5),
        (22, 80, 36, 29.0),
        (75, 97, 80, 11.0)
    ])]
    fn test_generate_segments(
        #[case] segmentation_type: SegmentationType,
        #[case] expected: Vec<(usize, usize, usize, f64)>,
    ) {
        let control = Control::default()
            .with_number_of_wild_segments(5)
            .with_minimal_relative_segment_length(0.2);
        let optimizer = testing::TrivialOptimizer { control: &control };
        let mut segmentation = Segmentation::new(segmentation_type, &optimizer);

        segmentation.generate_segments(&optimizer);
        assert_eq!(*segmentation.segments.borrow(), expected);
    }
}
