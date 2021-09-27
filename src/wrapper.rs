use crate::binary_segmentation::BinarySegmentationTree;
use crate::change_in_mean::ChangeInMean;
use crate::control::Control;
use ndarray;

pub fn hdcd(X: ndarray::Array2<f64>) -> Vec<usize> {
    let control = Control {
        minimal_gain_to_split: 0.1,
        minimal_relative_segment_length: 0.1,
    };
    let mut optimizer = ChangeInMean::new(&X);
    let mut binary_segmentation = BinarySegmentationTree::new(&X, control);

    binary_segmentation.grow(&mut optimizer);

    binary_segmentation.split_points()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::testing;

    #[test]
    fn test_binary_segmentation_wrapper() {
        let X = testing::array();

        assert_eq!(X.shape(), &[100, 5]);

        assert_eq!(hdcd(X), vec![25, 40, 80]);
    }
}
