use crate::{Control, Optimizer};
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
    n: usize,
    segmentation_type: SegmentationType,
    segments: Vec<(usize, usize, usize, f64)>,
    control: &'a Control,
}

impl<'a> Segmentation<'a> {
    pub fn new(segmentation_type: SegmentationType, n: usize, control: &'a Control) -> Self {
        Segmentation {
            n,
            segmentation_type,
            segments: vec![],
            control,
        }
    }

    pub fn generate_segments(&mut self, optimizer: &dyn Optimizer) {
        match self.segmentation_type {
            SegmentationType::BS => (),
            SegmentationType::SBS => {
                // See Definition 1 of https://arxiv.org/pdf/2002.06633.pdf
                let n_layers = ((2. * self.control.minimal_relative_segment_length).ln()
                    / self.control.seeded_segments_alpha.ln())
                .ceil();
                let mut segment_length: f64;
                let mut alpha_k: f64;
                let mut n_segments: f64;
                let mut segment_step: f64;
                let mut start: usize;
                let mut stop: usize;
                for k in 1..(n_layers as i32) {
                    alpha_k = self.control.seeded_segments_alpha.powi(k);
                    segment_length = (self.n as f64) * alpha_k;
                    n_segments = 2. * (1. / alpha_k).ceil() - 1.;
                    segment_step = (self.n as f64 - segment_length) / (n_segments - 1.);
                    for segment_id in 0..(n_segments as usize) {
                        start = (segment_id as f64 * segment_step) as usize;
                        stop = start + segment_length.ceil() as usize;
                        let (best_split, max_gain) =
                            optimizer.find_best_split(start, stop).unwrap();
                        self.segments.push((start, stop, best_split, max_gain));
                    }
                }
            }
            SegmentationType::WBS => {
                let mut rng = StdRng::seed_from_u64(self.control.seed as u64);
                let dist = Uniform::from(0..(self.n + 1));

                let mut start: usize;
                let mut stop: usize;

                while self.segments.len() < self.control.number_of_wild_segments {
                    start = dist.sample(&mut rng);
                    stop = dist.sample(&mut rng);
                    if start < stop {
                        if let Ok((best_split, max_gain)) = optimizer.find_best_split(start, stop) {
                            self.segments.push((start, stop, best_split, max_gain))
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::testing;
    use rstest::*;

    #[rstest]
    #[case(SegmentationType::BS, 10, vec![])]
    #[case(SegmentationType::SBS, 100, vec![
        (0, 71, 17, 35.5),
        (14, 85, 31, 35.5),
        (29, 100, 46, 35.5),
        (0, 50, 12, 25.0),
        (12, 62, 24, 25.0),
        (25, 75, 37, 25.0),
        (37, 87, 49, 25.0),
        (50, 100, 62, 25.0)
    ])]
    #[case(SegmentationType::WBS, 100, vec![
        (73, 78, 74, 2.5),
        (2, 59, 16, 28.5),
        (26, 77, 38, 25.5),
        (22, 80, 36, 29.0),
        (75, 97, 80, 11.0)
    ])]
    fn test_generate_segments(
        #[case] segmentation_type: SegmentationType,
        #[case] n: usize,
        #[case] expected: Vec<(usize, usize, usize, f64)>,
    ) {
        let control = Control::default()
            .with_number_of_wild_segments(5)
            .with_minimal_relative_segment_length(0.2);
        let mut segmentation = Segmentation::new(segmentation_type, n, &control);
        let optimizer = testing::TrivialOptimizer { control: &control };

        segmentation.generate_segments(&optimizer);
        assert_eq!(segmentation.segments, expected);
    }
}
