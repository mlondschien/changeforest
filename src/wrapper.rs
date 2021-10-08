use crate::binary_segmentation::BinarySegmentationTree;
use crate::classifier::kNN;
use crate::control::Control;
use crate::gain::ChangeInMean;
use crate::gain::ClassifierGain;
use crate::optimizer::GridSearch;
use ndarray;

pub fn hdcd(X: &ndarray::ArrayView2<'_, f64>) -> Vec<usize> {
    let control = Control {
        minimal_gain_to_split: 0.1,
        minimal_relative_segment_length: 0.1,
        alpha: 0.05,
    };
    let gain = ChangeInMean::new(X);
    let optimizer = GridSearch { gain };
    let mut binary_segmentation = BinarySegmentationTree::new(X, control);

    binary_segmentation.grow(&optimizer);

    binary_segmentation.split_points()
}

pub fn hdcd_knn(X: &ndarray::ArrayView2<'_, f64>) -> Vec<usize> {
    let control = Control {
        minimal_gain_to_split: 0.1,
        minimal_relative_segment_length: 0.1,
        alpha: 0.05,
    };

    let classifier = kNN::new(X);
    let gain = ClassifierGain { classifier };
    let optimizer = GridSearch { gain };
    let mut binary_segmentation = BinarySegmentationTree::new(X, control);

    binary_segmentation.grow(&optimizer);

    binary_segmentation.split_points()
}

#[cfg(test)]
mod tests {
    use crate::testing;

    #[test]
    fn test_binary_segmentation_wrapper() {
        let X = testing::array();

        assert_eq!(X.shape(), &[100, 5]);

        // TODO
        // assert_eq!(hdcd(&X.view()), vec![25, 40, 80]);
    }

    #[test]
    fn test_binary_segmentation_wrapper2() {
        let X = testing::array();

        assert_eq!(X.shape(), &[100, 5]);

        // TODO
        // assert_eq!(hdcd_knn(&X.view()), vec![11, 25, 40, 60, 80]);
    }
}
