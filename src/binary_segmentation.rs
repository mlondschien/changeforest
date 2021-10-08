use super::control::Control;
use super::model_selection::ModelSelection;
use crate::Optimizer;
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
    control: Control,
}

#[allow(dead_code)]
impl BinarySegmentationTree {
    pub fn new(X: &ndarray::ArrayView2<'_, f64>, control: Control) -> BinarySegmentationTree {
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

    pub fn grow<T: Optimizer>(&mut self, optimizer: &T) {
        if let Some(split_candidates) = self.split_candidates() {
            let result = optimizer.find_best_split(self.start, self.stop, split_candidates);
            if !result.is_significant {
                return;
            }

            let mut left = self.new_left(result.best_split);
            left.grow(optimizer);
            self.left = Some(left);

            let mut right = self.new_right(result.best_split);
            right.grow(optimizer);
            self.right = Some(right);

            self.split = Some(result.best_split);
        }
    }

    pub fn split_points(&self) -> Vec<usize> {
        if let Some(split_point) = self.split {
            let out = self.left.as_ref().unwrap().split_points().into_iter();
            let out = out.chain(vec![split_point].into_iter());
            let out = out.chain(self.right.as_ref().unwrap().split_points().into_iter());
            out.collect()
        } else {
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::control::Control;
    use super::*;
    use crate::optimizer::GridSearch;
    use crate::testing;

    #[test]
    fn test_binary_segmentation_change_in_mean() {
        let X = testing::array();
        let X_view = X.view();

        assert_eq!(X_view.shape(), &[100, 5]);

        let control = Control {
            minimal_gain_to_split: 0.1,
            minimal_relative_segment_length: 0.1,
            alpha: 0.05,
        };
        let mut gain = testing::ChangeInMean::new(&X_view);
        let mut optimizer = GridSearch { gain };
        let mut binary_segmentation = BinarySegmentationTree::new(&X_view, control);

        binary_segmentation.grow(&mut optimizer);

        assert_eq!(binary_segmentation.split, Some(25));
        assert_eq!(binary_segmentation.split_points(), vec![25, 40, 80]);
    }
}
