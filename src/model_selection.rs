use super::control::Control;
use super::gain::Gain;

pub trait ModelSelection: Gain {
    fn is_significant(&self, start: usize, stop: usize, split: usize, control: Control) -> bool {
        self.gain(start, stop, split) > control.minimal_gain_to_split
    }
}

#[cfg(test)]
mod tests {

    use super::super::control::Control;
    use super::super::testing::testing;
    use super::*;
    use rstest::*;

    #[rstest]
    #[case(0, 0, 0, 1., false)]
    #[case(0, 7, 2, 1., false)]
    #[case(0, 7, 2, 0., true)]
    #[case(0, 7, 2, 0.1, true)]
    #[case(0, 7, 0, 0.1, false)]
    fn test_change_in_mean_find_best_split(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] split: usize,
        #[case] minimal_gain_to_split: f64,
        #[case] expected: bool,
    ) {
        let X = ndarray::array![
            [0., 1.],
            [0., 1.],
            [1., -1.],
            [1., -1.],
            [-1., -1.],
            [-1., -1.],
            [-1., -1.]
        ];
        let X_view = X.view();
        assert_eq!(X.shape(), &[7, 2]);

        let change_in_mean = testing::ChangeInMean::new(&X_view);

        assert_eq!(
            change_in_mean.is_significant(
                start,
                stop,
                split,
                Control {
                    minimal_gain_to_split: minimal_gain_to_split,
                    minimal_relative_segment_length: 0.1,
                    alpha: 0.05
                }
            ),
            expected
        );
    }
}
