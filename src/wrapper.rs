use crate::binary_segmentation::BinarySegmentationTree;
use crate::classifier::kNN;
use crate::control::Control;
use crate::gain::{ChangeInMean, ClassifierGain};
use crate::optimizer::{GridSearch, TwoStepSearch};
use ndarray;

pub fn hdcd(X: &ndarray::ArrayView2<'_, f64>, method: &str) -> Vec<usize> {
    let control = Control::default();

    let mut binary_segmentation = BinarySegmentationTree::new(X, control);

    if method == "knn" {
        let classifier = kNN::new(X);
        let gain = ClassifierGain { classifier };
        let optimizer = GridSearch {
            gain,
            control: &control,
        };
        binary_segmentation.grow(&optimizer);
    } else if method == "change_in_mean" {
        let gain = ChangeInMean::new(X);
        let optimizer = TwoStepSearch {
            gain,
            control: &control,
        };
        binary_segmentation.grow(&optimizer);
    } else {
        panic!(
            "method should be one of 'knn' or 'change_in_mean'. Got {}",
            method
        );
    }
    binary_segmentation.split_points()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing;
    use rstest::*;

    #[rstest]
    #[case("knn")]
    #[case("change_in_mean")]
    fn test_binary_segmentation_wrapper(#[case] method: &str) {
        let X = testing::array();

        assert_eq!(X.shape(), &[100, 5]);
        assert_eq!(hdcd(&X.view(), method), vec![25, 40, 80]);
    }
}
