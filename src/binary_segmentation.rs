use crate::Segmentation;

#[allow(dead_code)]
pub struct BinarySegmentationTree {
    pub start: usize,
    pub stop: usize,
    pub n: usize,
    pub split: Option<usize>,
    pub max_gain: Option<f64>,
    pub is_significant: bool,
    pub left: Option<Box<BinarySegmentationTree>>,
    pub right: Option<Box<BinarySegmentationTree>>,
}

#[allow(dead_code)]
impl BinarySegmentationTree {
    pub fn new(X: &ndarray::ArrayView2<'_, f64>) -> BinarySegmentationTree {
        BinarySegmentationTree {
            start: 0,
            stop: X.nrows(),
            n: X.nrows(),
            split: Option::None,
            max_gain: Option::None,
            is_significant: false,
            left: Option::None,
            right: Option::None,
        }
    }

    fn new_left(&self, split: usize) -> Box<BinarySegmentationTree> {
        Box::new(BinarySegmentationTree {
            start: self.start,
            stop: split,
            n: self.n,
            split: Option::None,
            max_gain: Option::None,
            is_significant: false,
            left: Option::None,
            right: Option::None,
        })
    }

    fn new_right(&self, split: usize) -> Box<BinarySegmentationTree> {
        Box::new(BinarySegmentationTree {
            start: split,
            stop: self.stop,
            n: self.n,
            split: Option::None,
            max_gain: Option::None,
            is_significant: false,
            left: Option::None,
            right: Option::None,
        })
    }

    pub fn grow(&mut self, segmentation: &mut Segmentation) {
        if let Ok(optimizer_result) = segmentation.find_best_split(self.start, self.stop) {
            self.split = Some(optimizer_result.best_split);
            self.max_gain = Some(optimizer_result.max_gain);

            self.is_significant = segmentation.is_significant(
                self.start,
                self.stop,
                optimizer_result.best_split,
                optimizer_result.max_gain,
            );

            if !self.is_significant {
                return;
            }

            let mut left = self.new_left(optimizer_result.best_split);
            left.grow(segmentation);
            self.left = Some(left);

            let mut right = self.new_right(optimizer_result.best_split);
            right.grow(segmentation);
            self.right = Some(right);
        }
    }
}

pub struct BinarySegmentationResult {
    pub start: usize,
    pub stop: usize,
    pub best_split: Option<usize>,
    pub max_gain: Option<f64>,
    pub is_significant: bool,
    pub left: Option<Box<BinarySegmentationResult>>,
    pub right: Option<Box<BinarySegmentationResult>>,
}

impl BinarySegmentationResult {
    pub fn from_tree(tree: &BinarySegmentationTree) -> Self {
        let left = tree
            .left
            .as_ref()
            .map(|tree| Box::new(BinarySegmentationResult::from_tree(tree)));

        let right = tree
            .right
            .as_ref()
            .map(|tree| Box::new(BinarySegmentationResult::from_tree(tree)));

        BinarySegmentationResult {
            start: tree.start,
            stop: tree.stop,
            best_split: tree.split,
            max_gain: tree.max_gain,
            is_significant: tree.is_significant,
            left,
            right,
        }
    }

    pub fn split_points(&self) -> Vec<usize> {
        let mut split_points = vec![];

        if let Some(left_boxed) = &self.left {
            split_points.append(&mut left_boxed.split_points());
        }

        if let Some(best_split) = self.best_split {
            if self.is_significant {
                split_points.push(best_split);
            }
        }

        if let Some(right_boxed) = &self.right {
            split_points.append(&mut right_boxed.split_points());
        }

        split_points
    }
}

#[cfg(test)]
mod tests {
    use super::super::control::Control;
    use super::*;
    use crate::optimizer::GridSearch;
    use crate::segmentation::{Segmentation, SegmentationType};
    use crate::testing;

    #[test]
    fn test_binary_segmentation_change_in_mean() {
        let X = testing::array();
        let X_view = X.view();

        assert_eq!(X_view.shape(), &[100, 5]);

        let control = Control::default();
        let gain = testing::ChangeInMean::new(&X_view);
        let optimizer = GridSearch {
            gain,
            control: &control,
        };
        let mut segmentation = Segmentation::new(SegmentationType::BS, &optimizer);
        let mut binary_segmentation = BinarySegmentationTree::new(&X_view);

        binary_segmentation.grow(&mut segmentation);

        assert_eq!(binary_segmentation.split, Some(25));
    }

    #[test]
    fn test_binary_segmentation_result() {
        let X = testing::array();
        let X_view = X.view();

        assert_eq!(X_view.shape(), &[100, 5]);

        let control = Control::default();
        let gain = testing::ChangeInMean::new(&X_view);
        let optimizer = GridSearch {
            gain,
            control: &control,
        };
        let mut segmentation = Segmentation::new(SegmentationType::BS, &optimizer);
        let mut tree = BinarySegmentationTree::new(&X_view);

        tree.grow(&mut segmentation);

        let result = BinarySegmentationResult::from_tree(&tree);

        assert_eq!(result.split_points(), vec![25, 40, 80]);
        assert_eq!(result.start, 0);
        assert_eq!(result.stop, 100);
        assert_eq!(result.best_split, Some(25));
        assert_eq!(result.is_significant, true);

        let right = result.right.unwrap();
        assert_eq!(right.split_points(), vec![40, 80]);
        assert_eq!(right.start, 25);
        assert_eq!(right.stop, 100);
        assert_eq!(right.best_split, Some(40));
        assert_eq!(right.is_significant, true);

        let left = result.left.unwrap();
        assert_eq!(left.split_points(), vec![]);
        assert_eq!(left.start, 0);
        assert_eq!(left.stop, 25);
        assert_eq!(left.best_split, Some(10));
        assert_eq!(left.is_significant, false);
    }
}
