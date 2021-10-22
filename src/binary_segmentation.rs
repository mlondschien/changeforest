use crate::Optimizer;

#[allow(dead_code)]
pub struct BinarySegmentationTree<'a> {
    pub start: usize,
    pub stop: usize,
    pub n: usize,
    pub split: Option<usize>,
    pub max_gain: Option<f64>,
    pub is_significant: bool,
    pub left: Option<Box<BinarySegmentationTree<'a>>>,
    pub right: Option<Box<BinarySegmentationTree<'a>>>,
    optimizer: &'a dyn Optimizer,
}

#[allow(dead_code)]
impl<'a> BinarySegmentationTree<'a> {
    pub fn new(
        X: &ndarray::ArrayView2<'_, f64>,
        optimizer: &'a impl Optimizer,
    ) -> BinarySegmentationTree<'a> {
        BinarySegmentationTree {
            start: 0,
            stop: X.nrows(),
            n: X.nrows(),
            split: Option::None,
            max_gain: Option::None,
            is_significant: false,
            left: Option::None,
            right: Option::None,
            optimizer,
        }
    }

    fn new_left(&self, split: usize) -> Box<BinarySegmentationTree<'a>> {
        Box::new(BinarySegmentationTree {
            start: self.start,
            stop: split,
            n: self.n,
            split: Option::None,
            max_gain: Option::None,
            is_significant: false,
            left: Option::None,
            right: Option::None,
            optimizer: self.optimizer,
        })
    }

    fn new_right(&self, split: usize) -> Box<BinarySegmentationTree<'a>> {
        Box::new(BinarySegmentationTree {
            start: split,
            stop: self.stop,
            n: self.n,
            split: Option::None,
            max_gain: Option::None,
            is_significant: false,
            left: Option::None,
            right: Option::None,
            optimizer: self.optimizer,
        })
    }

    pub fn grow(&mut self) {
        if let Ok((best_split, max_gain)) = self.optimizer.find_best_split(self.start, self.stop) {
            self.split = Some(best_split);
            self.max_gain = Some(max_gain);

            self.is_significant = self
                .optimizer
                .is_significant(self.start, self.stop, best_split, max_gain);

            if !self.is_significant {
                return;
            }

            let mut left = self.new_left(best_split);
            left.grow();
            self.left = Some(left);

            let mut right = self.new_right(best_split);
            right.grow();
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
        let mut binary_segmentation = BinarySegmentationTree::new(&X_view, &optimizer);

        binary_segmentation.grow();

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
        let mut tree = BinarySegmentationTree::new(&X_view, &optimizer);

        tree.grow();

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
