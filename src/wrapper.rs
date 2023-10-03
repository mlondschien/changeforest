use crate::classifier::{kNN, RandomForest};
use crate::control::Control;
use crate::gain::{ChangeInMean, ClassifierGain};
use crate::optimizer::{GridSearch, TwoStepSearch};
use crate::segmentation::{Segmentation, SegmentationType};
use crate::{BinarySegmentationResult, BinarySegmentationTree};
use ndarray;

pub fn changeforest(
    X: &ndarray::ArrayView2<'_, f64>,
    method: &str,
    segmentation_type: &str,
    control: &Control,
) -> BinarySegmentationResult {
    let segmentation_type_enum: SegmentationType;
    let mut tree: BinarySegmentationTree;

    if segmentation_type == "bs" {
        segmentation_type_enum = SegmentationType::BS;
    } else if segmentation_type == "sbs" {
        segmentation_type_enum = SegmentationType::SBS;
    } else if segmentation_type == "wbs" {
        segmentation_type_enum = SegmentationType::WBS;
    } else {
        panic!("segmentation_type must be one of 'bs', 'sbs', 'wbs'")
    }

    if method == "knn" {
        let classifier = kNN::new(X, control);
        let gain = ClassifierGain { classifier };
        let optimizer = TwoStepSearch { gain };
        let mut segmentation = Segmentation::new(segmentation_type_enum, &optimizer);
        tree = BinarySegmentationTree::new(X);
        tree.grow(&mut segmentation);
        BinarySegmentationResult::from_tree(tree).with_segments(segmentation)
    } else if method == "random_forest" {
        let classifier = RandomForest::new(X, control);
        let gain = ClassifierGain { classifier };
        let optimizer = TwoStepSearch { gain };
        let mut segmentation = Segmentation::new(segmentation_type_enum, &optimizer);
        tree = BinarySegmentationTree::new(X);
        tree.grow(&mut segmentation);
        BinarySegmentationResult::from_tree(tree).with_segments(segmentation)
    } else if method == "change_in_mean" {
        let gain = ChangeInMean::new(X, control);
        let optimizer = GridSearch { gain };
        let mut segmentation = Segmentation::new(segmentation_type_enum, &optimizer);
        tree = BinarySegmentationTree::new(X);
        tree.grow(&mut segmentation);
        BinarySegmentationResult::from_tree(tree).with_segments(segmentation)
    } else {
        panic!("method should be one of 'knn', 'random_forest' or 'change_in_mean'. Got {method}",);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;
    use rstest::*;

    #[rstest]
    #[case("knn", "bs")]
    #[case("knn", "wbs")]
    #[case("knn", "sbs")]
    #[case("change_in_mean", "bs")]
    #[case("change_in_mean", "wbs")]
    #[case("change_in_mean", "sbs")]
    #[case("random_forest", "bs")]
    //#[case("random_forest", "wbs")]
    #[case("random_forest", "sbs")]
    fn test_binary_segmentation_wrapper(#[case] method: &str, #[case] segmentation_type: &str) {
        let X = testing::array();
        let control = Control::default().with_minimal_relative_segment_length(0.1);

        assert_eq!(X.shape(), &[100, 5]);
        assert_eq!(
            changeforest(&X.view(), method, segmentation_type, &control).split_points(),
            vec![25, 40, 80]
        );
    }
}
