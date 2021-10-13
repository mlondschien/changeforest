use crate::binary_segmentation::BinarySegmentationTree;
use crate::classifier::kNN;
use crate::control::Control;
use crate::gain::{ChangeInMean, ClassifierGain};
use crate::optimizer::{GridSearch, TwoStepSearch};
use crate::segmentation::{Segmentation, SegmentationType};
use ndarray;

pub fn hdcd(
    X: &ndarray::ArrayView2<'_, f64>,
    method: &str,
    segmentation_type: SegmentationType,
) -> Vec<usize> {
    let control = Control::default();

    if method == "knn" {
        let classifier = kNN::new(X);
        let gain = ClassifierGain { classifier };
        let optimizer = GridSearch {
            gain,
            control: &control,
        };
        let segmentation = Segmentation::new(segmentation_type, &optimizer);
        let mut binary_segmentation = BinarySegmentationTree::new(X, &segmentation);
        binary_segmentation.grow();
        binary_segmentation.split_points()
    } else if method == "change_in_mean" {
        let gain = ChangeInMean::new(X);
        let optimizer = TwoStepSearch {
            gain,
            control: &control,
        };
        let segmentation = Segmentation::new(segmentation_type, &optimizer);
        let mut binary_segmentation = BinarySegmentationTree::new(X, &segmentation);
        binary_segmentation.grow();
        binary_segmentation.split_points()
    } else {
        panic!(
            "method should be one of 'knn' or 'change_in_mean'. Got {}",
            method
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;
    use rstest::*;

    #[rstest]
    #[case("knn", SegmentationType::BS)]
    #[case("knn", SegmentationType::WBS)]
    #[case("knn", SegmentationType::SBS)]
    #[case("change_in_mean", SegmentationType::BS)]
    #[case("change_in_mean", SegmentationType::WBS)]
    #[case("change_in_mean", SegmentationType::SBS)]
    fn test_binary_segmentation_wrapper(
        #[case] method: &str,
        #[case] segmentation_type: SegmentationType,
    ) {
        let X = testing::array();

        assert_eq!(X.shape(), &[100, 5]);
        assert_eq!(hdcd(&X.view(), method, segmentation_type), vec![25, 40, 80]);
    }
}
