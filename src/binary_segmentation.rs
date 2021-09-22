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
            control: control,
            left: Option::None,
            right: Option::None,
        }
    }

    fn grow(&mut self, optimizer: &mut impl optimizer::Optimizer) {
        if self.control.minimal_relative_segment_length * (self.n as f64)
            >= 2. * (self.stop - self.start) as f64
        {
            return ();
        }

        let best_split = optimizer.find_best_split(self.start, self.stop);

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

        let mut optimizer = optimizer::ChangeInMean::new(&X);
        let control = control::Control {
            minimal_relative_segment_length: 0.1,
        };
        let mut binary_segmentation = BinarySegmentationTree::new(&X, control);

        binary_segmentation.grow(&mut optimizer);

        assert_eq!(binary_segmentation.split, Some(25));
        assert_eq!(binary_segmentation.right.unwrap().split, Some(40));
    }
}
