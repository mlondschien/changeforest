use crate::binary_segmentation::BinarySegmentationTree;
use crate::classifier::kNN;
use crate::control::Control;
use crate::gain::{ChangeInMean, ClassifierGain};
use crate::optimizer::{GridSearch, TwoStepSearch};
use crate::segmentation::{Segmentation, SegmentationType};
use ndarray;

pub fn hdcd(X: &ndarray::ArrayView2<'_, f64>, method: &str, segmentation_type: &str) -> Vec<usize> {
    let control = Control::default();

    let segmentation_type_enum: SegmentationType;

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
        let classifier = kNN::new(X);
        let gain = ClassifierGain { classifier };
        let optimizer = GridSearch {
            gain,
            control: &control,
        };
        let segmentation = Segmentation::new(segmentation_type_enum, &optimizer);
        let mut binary_segmentation = BinarySegmentationTree::new(X, &segmentation);
        binary_segmentation.grow();
        binary_segmentation.split_points()
    } else if method == "change_in_mean" {
        let gain = ChangeInMean::new(X);
        let optimizer = TwoStepSearch {
            gain,
            control: &control,
        };
        let segmentation = Segmentation::new(segmentation_type_enum, &optimizer);
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
    #[case("knn", "bs")]
    #[case("knn", "wbs")]
    #[case("knn", "sbs")]
    #[case("change_in_mean", "bs")]
    #[case("change_in_mean", "wbs")]
    #[case("change_in_mean", "sbs")]
    fn test_binary_segmentation_wrapper(#[case] method: &str, #[case] segmentation_type: &str) {
        let X = testing::array();

        assert_eq!(X.shape(), &[100, 5]);
        assert_eq!(hdcd(&X.view(), method, segmentation_type), vec![25, 40, 80]);
    }
}
