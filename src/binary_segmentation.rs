use crate::optimizer::OptimizerResult;
use crate::{ModelSelectionResult, Segmentation};

pub struct BinarySegmentationTree {
    pub start: usize,
    pub stop: usize,
    pub n: usize,
    pub model_selection_result: ModelSelectionResult,
    pub left: Option<Box<BinarySegmentationTree>>,
    pub right: Option<Box<BinarySegmentationTree>>,
    pub optimizer_result: Option<OptimizerResult>,
}

impl BinarySegmentationTree {
    pub fn new(X: &ndarray::ArrayView2<'_, f64>) -> BinarySegmentationTree {
        BinarySegmentationTree {
            start: 0,
            stop: X.nrows(),
            n: X.nrows(),
            model_selection_result: ModelSelectionResult::default(),
            left: None,
            right: None,
            optimizer_result: None,
        }
    }

    fn new_left(&self, split: usize) -> Box<BinarySegmentationTree> {
        Box::new(BinarySegmentationTree {
            start: self.start,
            stop: split,
            n: self.n,
            model_selection_result: ModelSelectionResult::default(),
            left: None,
            right: None,
            optimizer_result: None,
        })
    }

    fn new_right(&self, split: usize) -> Box<BinarySegmentationTree> {
        Box::new(BinarySegmentationTree {
            start: split,
            stop: self.stop,
            n: self.n,
            model_selection_result: ModelSelectionResult::default(),
            left: None,
            right: None,
            optimizer_result: None,
        })
    }

    /// Grow a `BinarySegmentationTree`.
    ///
    /// Recursively split segments and add subsegments as children `left` and
    /// `right` until segments are smaller then the minimal segment length
    /// (`n * control.minimal_relative_segment_length`) or the `OptimizerResult` is no
    /// longer significant.
    pub fn grow(&mut self, segmentation: &mut Segmentation) {
        if let Ok(optimizer_result) = segmentation.find_best_split(self.start, self.stop) {
            self.model_selection_result = segmentation.model_selection(&optimizer_result);

            if self.model_selection_result.is_significant {
                let mut left = self.new_left(optimizer_result.best_split);
                left.grow(segmentation);
                self.left = Some(left);

                let mut right = self.new_right(optimizer_result.best_split);
                right.grow(segmentation);
                self.right = Some(right);
            }

            self.optimizer_result = Some(optimizer_result);
        }
    }
}

#[derive(Clone, Debug)]
/// Struct holding results from a BinarySegmentationTree after fitting.
pub struct BinarySegmentationResult {
    pub start: usize,
    pub stop: usize,
    pub model_selection_result: ModelSelectionResult,
    pub optimizer_result: Option<OptimizerResult>,
    pub left: Option<Box<BinarySegmentationResult>>,
    pub right: Option<Box<BinarySegmentationResult>>,
    pub segments: Option<Vec<OptimizerResult>>,
}

impl BinarySegmentationResult {
    pub fn from_tree(tree: BinarySegmentationTree) -> Self {
        let left = tree
            .left
            .map(|tree| Box::new(BinarySegmentationResult::from_tree(*tree)));

        let right = tree
            .right
            .map(|tree| Box::new(BinarySegmentationResult::from_tree(*tree)));

        BinarySegmentationResult {
            start: tree.start,
            stop: tree.stop,
            model_selection_result: tree.model_selection_result,
            optimizer_result: tree.optimizer_result,
            left,
            right,
            segments: None,
        }
    }

    pub fn split_points(&self) -> Vec<usize> {
        let mut split_points = vec![];

        if let Some(left_boxed) = &self.left {
            split_points.append(&mut left_boxed.split_points());
        }

        if let Some(result) = self.optimizer_result.as_ref() {
            if self.model_selection_result.is_significant {
                split_points.push(result.best_split);
            }
        }

        if let Some(right_boxed) = &self.right {
            split_points.append(&mut right_boxed.split_points());
        }

        split_points
    }

    pub fn with_segments(mut self, segmentation: Segmentation) -> Self {
        self.segments = Some(segmentation.segments);
        self
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
        let gain = testing::ChangeInMean::new(&X_view, &control);
        let optimizer = GridSearch { gain };
        let mut segmentation = Segmentation::new(SegmentationType::BS, &optimizer);
        let mut binary_segmentation = BinarySegmentationTree::new(&X_view);

        binary_segmentation.grow(&mut segmentation);

        let optimizer_result = binary_segmentation.optimizer_result.as_ref().unwrap();
        assert_eq!(optimizer_result.best_split, 80);
        assert_eq!(optimizer_result.start, 0);
        assert_eq!(optimizer_result.stop, 100);
    }

    #[test]
    fn test_binary_segmentation_result() {
        let X = testing::array();
        let X_view = X.view();

        assert_eq!(X_view.shape(), &[100, 5]);
        let control = Control::default();
        let gain = testing::ChangeInMean::new(&X_view, &control);
        let optimizer = GridSearch { gain };
        let mut segmentation = Segmentation::new(SegmentationType::SBS, &optimizer);
        let mut tree = BinarySegmentationTree::new(&X_view);

        tree.grow(&mut segmentation);

        let result = BinarySegmentationResult::from_tree(tree);

        assert_eq!(result.split_points(), vec![25, 40, 80]);
        assert_eq!(result.start, 0);
        assert_eq!(result.stop, 100);
        assert_eq!(result.optimizer_result.as_ref().unwrap().best_split, 40);
        assert!(result.model_selection_result.is_significant);
        assert!(result.optimizer_result.is_some());

        let right = result.right.as_ref().unwrap();
        assert_eq!(right.split_points(), vec![80]);
        assert_eq!(right.start, 40);
        assert_eq!(right.stop, 100);
        assert_eq!(right.optimizer_result.as_ref().unwrap().best_split, 80);
        assert!(right.model_selection_result.is_significant);
        assert!(right.optimizer_result.is_some());

        let left = result.left.as_ref().unwrap();
        assert_eq!(left.split_points(), vec![25]);
        assert_eq!(left.start, 0);
        assert_eq!(left.stop, 40);
        assert_eq!(left.optimizer_result.as_ref().unwrap().best_split, 25);
        assert!(left.model_selection_result.is_significant);
        assert!(left.optimizer_result.is_some());

        let result = result.with_segments(segmentation);
        assert!(!result.segments.as_ref().unwrap().is_empty());
    }
}
