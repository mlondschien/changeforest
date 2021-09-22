use super::control;
use super::optimizer;
use ndarray;
use std;

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

    fn split_candidates(&self) -> Option<std::ops::Range<usize>> {
        let minimal_segment_length =
            (self.control.minimal_relative_segment_length * (self.n as f64)).round() as usize;

        if 2 * minimal_segment_length >= (self.stop - self.start) {
            None
        } else {
            Some((self.start + minimal_segment_length)..(self.stop - minimal_segment_length))
        }
    }

    fn new_left(&self, split: usize) -> Box<BinarySegmentationTree> {
        Box::new(BinarySegmentationTree {
            start: self.start,
            stop: split,
            n: self.n,
            split: Option::None,
            left: Option::None,
            right: Option::None,
            control: self.control,
        })
    }

    fn new_right(&self, split: usize) -> Box<BinarySegmentationTree> {
        Box::new(BinarySegmentationTree {
            start: split,
            stop: self.stop,
            n: self.n,
            split: Option::None,
            left: Option::None,
            right: Option::None,
            control: self.control,
        })
    }

    fn grow(&mut self, optimizer: &mut impl optimizer::Optimizer) {
        if let Some(split_candidates) = self.split_candidates() {
            let best_split = optimizer.find_best_split(self.start, self.stop, split_candidates);

            let mut left = self.new_left(best_split);
            left.grow(optimizer);
            self.left = Some(left);

            let mut right = self.new_right(best_split);
            right.grow(optimizer);
            self.right = Some(right);

            self.split = Some(best_split);
        }
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
    use super::super::testing::testing;
    use super::*;

    #[test]
    fn test_binary_segmentation_change_in_mean() {
        let X = testing::array();

        assert_eq!(X.shape(), &[100, 5]);

        let control = control::Control {
            minimal_relative_segment_length: 0.1,
        };
        let mut optimizer = testing::ChangeInMean::new(&X);
        let mut binary_segmentation = BinarySegmentationTree::new(&X, control);

        binary_segmentation.grow(&mut optimizer);

        assert_eq!(binary_segmentation.split, Some(25));
        assert_eq!(
            binary_segmentation.split_points(),
            vec![14, 25, 40, 56, 69, 80]
        );
    }
}
