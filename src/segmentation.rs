use crate::control::Control;
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

pub struct Segmentation {
    n: usize,
    segmentation_type: SegmentationType,
    segments: Vec<(usize, usize)>,
    gains: Vec<(usize, usize, usize, f64)>,
}

impl Segmentation {
    pub fn new(segmentation_type: SegmentationType, n: usize) -> Self {
        Segmentation {
            n,
            segmentation_type,
            segments: vec![],
            gains: vec![],
        }
    }

    pub fn generate_segments(&mut self, control: &Control) {
        self.segments.push((0, self.n));
        match self.segmentation_type {
            SegmentationType::BS => (),
            SegmentationType::SBS => {
                // See Definition 1 of https://arxiv.org/pdf/2002.06633.pdf
                let n_layers = ((2. * control.minimal_relative_segment_length).ln()
                    / control.seeded_segments_alpha.ln())
                .ceil();
                let mut segment_length: f64;
                let mut alpha_k: f64;
                let mut n_segments: f64;
                let mut segment_step: f64;
                let mut start: usize;
                for k in 1..(n_layers as i32) {
                    alpha_k = control.seeded_segments_alpha.powi(k);
                    segment_length = (self.n as f64) * alpha_k;
                    n_segments = 2. * (1. / alpha_k).ceil() - 1.;
                    segment_step = (self.n as f64 - segment_length) / (n_segments - 1.);
                    println!(
                        "alpha_k={}, segment_length={}, n_segments={}, segment_step={}",
                        alpha_k, segment_length, n_segments, segment_step
                    );
                    for segment_id in 0..(n_segments as usize) {
                        start = (segment_id as f64 * segment_step) as usize;
                        println!(
                            "segment_id={}, start={}, stop={}",
                            segment_id,
                            start,
                            start + segment_length.ceil() as usize
                        );
                        self.segments
                            .push((start, start + segment_length.ceil() as usize));
                    }
                }
            }
            SegmentationType::WBS => {
                let mut rng = StdRng::seed_from_u64(control.seed as u64);
                let dist = Uniform::from(0..(self.n + 1));
                let mut left: usize;
                let mut right: usize;
                while self.segments.len() < control.number_of_wild_segments {
                    left = dist.sample(&mut rng);
                    right = dist.sample(&mut rng);
                    if left < right && !self.segments.contains(&(left, right)) {
                        self.segments.push((left, right))
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use rstest::*;

    #[rstest]
    #[case(10, SegmentationType::BS, vec![(0, 10)])]
    #[case(100, SegmentationType::SBS, vec![(0, 100), (0, 71), (14, 85), (29, 100), (0, 50), (12, 62), (25, 75), (37, 87), (50, 100)])]
    #[case(10, SegmentationType::WBS, vec![(0, 10), (0, 6), (2, 8), (8, 10), (1, 10)])]
    fn test_segmentation(
        #[case] n: usize,
        #[case] segmentation_type: SegmentationType,
        #[case] expected: Vec<(usize, usize)>,
    ) {
        let mut control = Control::default();
        control
            .with_number_of_wild_segments(5)
            .with_minimal_relative_segment_length(0.2);
        let mut segmentation = Segmentation::new(segmentation_type, n);
        segmentation.generate_segments(&control);

        assert_eq!(segmentation.segments, expected);
    }
}
