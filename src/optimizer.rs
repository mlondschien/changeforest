use super::gain;

pub trait Optimizer: gain::Gain {
    fn find_best_split(
        &mut self,
        start: usize,
        stop: usize,
        split_candidates: impl Iterator<Item = usize>,
    ) -> usize {
        let mut max_index = 0;
        let mut max_value = -f64::INFINITY;
        let mut gain: f64;

        for index in split_candidates {
            gain = self.gain(start, stop, index);
            if gain > max_value {
                max_index = index;
                max_value = gain;
            }
        }
        max_index
    }
}

#[cfg(test)]
mod tests {

    use super::super::testing::testing;
    use super::*;
    use rstest::*;

    #[rstest]
    #[case(0, 0, 0)]
    #[case(0, 7, 2)]
    #[case(1, 7, 4)]
    #[case(2, 7, 4)]
    #[case(3, 7, 4)]
    #[case(1, 5, 2)]
    #[case(1, 6, 4)]
    #[case(1, 7, 4)]
    #[case(2, 6, 4)]
    #[case(2, 7, 4)]
    #[case(3, 6, 4)]
    fn test_change_in_mean_find_best_split(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] expected: usize,
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
        assert_eq!(X_view.shape(), &[7, 2]);

        let mut change_in_mean = testing::ChangeInMean::new(&X_view);

        assert_eq!(
            change_in_mean.find_best_split(start, stop, start..stop),
            expected
        );
    }
}
