pub trait Gain {
    #[allow(unused_variables)]
    fn loss(&self, start: usize, stop: usize) -> f64 {
        panic!("Not implemented.");
    }

    fn gain(&self, start: usize, stop: usize, split: usize) -> f64 {
        self.loss(start, stop) - self.loss(start, split) - self.loss(split, stop)
    }

    fn n(&self) -> usize;

    fn gain_full(
        &self,
        start: usize,
        stop: usize,
        split_points: Vec<usize>,
    ) -> ndarray::Array1<f64> {
        let mut gain = ndarray::Array::from_elem(self.n(), f64::NAN);

        for split_point in split_points {
            gain[split_point] = self.gain(start, stop, split_point);
        }

        gain
    }

    #[allow(unused_variables)]
    fn gain_approx(
        &self,
        start: usize,
        stop: usize,
        guess: usize,
        split_points: Vec<usize>,
    ) -> ndarray::Array1<f64> {
        self.gain_full(start, stop, split_points)
    }
}

#[cfg(test)]
mod tests {

    use super::super::testing::testing;
    use super::*;
    use assert_approx_eq::*;
    use rstest::*;

    #[rstest]
    #[case(0, 4, 0.25)]
    #[case(0, 2, 0.)]
    #[case(0, 3, 1. / 6.)]
    #[case(1, 4, 1. / 6.)]
    #[case(1, 3, 0.125)]
    #[case(3, 3, 0.)]
    fn test_change_in_mean_loss(#[case] start: usize, #[case] stop: usize, #[case] expected: f64) {
        let X = ndarray::array![[0., 0.], [0., 0.], [0., 1.], [0., 1.]];
        let X_view = X.view();

        assert_eq!(X.shape(), &[4, 2]);

        let change_in_mean = testing::ChangeInMean::new(&X_view);
        assert_approx_eq!(change_in_mean.loss(start, stop), expected);
    }

    #[rstest]
    #[case(0, 4, 2, 1. / 6.)]
    #[case(0, 4, 0, 0.)]
    #[case(0, 4, 1, 1. / 18.)]
    #[case(0, 4, 3, 1. / 18.)]
    #[case(0, 3, 2, 1. / 9.)]
    #[case(0, 3, 1, 1. / 36.)]
    #[case(0, 6, 0, 0.)]
    #[case(0, 6, 1, 1. / 90.)]
    #[case(0, 6, 2, 1. / 36.)]
    #[case(0, 6, 3, 1. / 18.)]
    #[case(0, 6, 4, 5. / 18.)]
    #[case(0, 6, 5, 1. / 90.)]
    fn test_change_in_mean_gain(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] split: usize,
        #[case] expected: f64,
    ) {
        let X = ndarray::array![[1., 0.], [1., 0.], [1., 1.], [1., 1.], [0., -1.], [1., 0.]];
        let X_view = X.view();
        assert_eq!(X_view.shape(), &[6, 2]);

        let change_in_mean = testing::ChangeInMean::new(&X_view);
        assert_approx_eq!(change_in_mean.gain(start, stop, split), expected);
        assert_approx_eq!(
            change_in_mean.gain_full(start, stop, vec![split])[split],
            expected
        );
        assert_approx_eq!(
            change_in_mean.gain_approx(start, stop, split, vec![split])[split],
            expected
        );
    }
}
