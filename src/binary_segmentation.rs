use super::control;
use super::optimizer;
use ndarray;

#[allow(dead_code)]
pub struct BinarySegmentationTree {
    start: usize,
    stop: usize,
    n: usize,
    split: Option<usize>,
    left: Option<Box<BinarySegmentationTree>>,
    right: Option<Box<BinarySegmentationTree>>,
    control: control::Control,
}

#[allow(dead_code)]
impl BinarySegmentationTree {
    fn new(X: &ndarray::Array2<f64>, control: control::Control) -> BinarySegmentationTree {
        BinarySegmentationTree {
            start: 0,
            stop: X.nrows(),
            n: X.nrows(),
            split: Option::None,
            control,
            left: Option::None,
            right: Option::None,
        }
    }

    fn grow(&mut self, optimizer: &mut impl optimizer::Optimizer) {
        let minimal_segment_length =
            (self.control.minimal_relative_segment_length * (self.n as f64)).round() as usize;

        if 2 * minimal_segment_length >= (self.stop - self.start) {
            return;
        }

        let split_candidates =
            (self.start + minimal_segment_length)..(self.stop - minimal_segment_length);

        assert!(!split_candidates.is_empty());

        let best_split = optimizer.find_best_split(self.start, self.stop, split_candidates);

        let mut left = Box::new(BinarySegmentationTree {
            start: self.start,
            stop: best_split,
            n: self.n,
            split: Option::None,
            left: Option::None,
            right: Option::None,
            control: self.control,
        });
        let mut right = Box::new(BinarySegmentationTree {
            start: best_split,
            stop: self.stop,
            n: self.n,
            split: Option::None,
            left: Option::None,
            right: Option::None,
            control: self.control,
        });

        left.grow(optimizer);
        right.grow(optimizer);

        self.split = Some(best_split);
        self.left = Some(left);
        self.right = Some(right);
    }

    fn split_points(&self) -> Vec<usize> {
        if let Some(split_point) = self.split {
            let out = self.left.as_ref().unwrap().split_points().into_iter();
            let out = out.chain(vec![split_point].into_iter());
            let out = out.chain(self.right.as_ref().unwrap().split_points().into_iter());
            out.collect()
        } else {
            return vec![];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::optimizer;
    use super::*;
    use ndarray::{s, Array};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_binary_segmentation_change_in_mean() {
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut X = Array::zeros((100, 5)); //

        X.slice_mut(s![0..25, 0]).fill(2.);
        X.slice_mut(s![40..80, 0]).fill(1.);
        X.slice_mut(s![0..40, 1]).fill(-2.);
        X.slice_mut(s![40..100, 1]).fill(-3.);

        let X = X + Array::random_using((100, 5), Uniform::new(0., 1.), &mut rng);

        assert_eq!(X.shape(), &[100, 5]);

        let control = control::Control {
            minimal_relative_segment_length: 0.1,
        };
        let mut optimizer = optimizer::ChangeInMean::new(&X);
        let mut binary_segmentation = BinarySegmentationTree::new(&X, control);

        binary_segmentation.grow(&mut optimizer);

        assert_eq!(binary_segmentation.split, Some(25));
        assert_eq!(
            binary_segmentation.split_points(),
            vec![14, 25, 40, 56, 69, 80]
        );
    }
}
