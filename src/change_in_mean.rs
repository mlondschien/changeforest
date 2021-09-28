use super::gain::Gain;
use super::model_selection::ModelSelection;
use super::optimizer::Optimizer;

pub struct ChangeInMean<'a> {
    X: &'a ndarray::ArrayView2<'a, f64>,
    X_cumsum: Option<ndarray::Array2<f64>>,
}

impl<'a> ChangeInMean<'a> {
    #[allow(dead_code)]
    pub fn new(X: &'a ndarray::ArrayView2<'a, f64>) -> ChangeInMean<'a> {
        ChangeInMean {
            X,
            X_cumsum: Option::None,
        }
    }

    fn calculate_cumsum(&mut self) {
        let mut X_cumsum = ndarray::Array2::zeros((self.X.nrows() + 1, self.X.ncols()));
        let mut slice = X_cumsum.slice_mut(ndarray::s![1.., ..]);
        slice += &self.X.view();

        X_cumsum.accumulate_axis_inplace(ndarray::Axis(0), |&prev, curr| *curr += prev);
        self.X_cumsum = Some(X_cumsum);
    }
}

impl<'a> Gain for ChangeInMean<'a> {
    fn n(&self) -> usize {
        self.X.nrows()
    }

    fn gain(&mut self, start: usize, stop: usize, split: usize) -> f64 {
        if (start == split) | (split == stop) {
            return 0.;
        }

        if self.X_cumsum.is_none() {
            self.calculate_cumsum();
        }

        let X_cumsum = self.X_cumsum.as_ref().unwrap();

        let s_1 = (split - start) as f64;
        let s_2 = (stop - split) as f64;
        let s = s_1 + s_2;

        let mut result = 0.;
        for idx in 0..self.X.ncols() {
            result += (s_1 * X_cumsum[[stop, idx]] + s_2 * X_cumsum[[start, idx]]
                - s * X_cumsum[[split, idx]])
            .powi(2)
        }
        result / (s * s_1 * s_2 * (self.X.nrows() as f64))
    }
}

impl<'a> Optimizer for ChangeInMean<'a> {}
impl<'a> ModelSelection for ChangeInMean<'a> {}

#[cfg(test)]
mod tests {

    use super::super::testing::testing;
    use super::*;
    use assert_approx_eq::*;
    use rstest::*;

    #[test]
    fn test_X_cumsum() {
        let X = ndarray::array![[1., 0.], [1., 0.], [1., 1.], [1., 1.]];
        let X_view = X.view();

        let mut change_in_mean = ChangeInMean::new(&X_view);
        change_in_mean.calculate_cumsum();

        let expected = ndarray::array![[0., 0.], [1., 0.], [2., 0.], [3., 1.], [4., 2.]];
        assert_eq!(change_in_mean.X_cumsum.unwrap(), expected);
    }

    #[rstest]
    #[case(0, 100)]
    #[case(0, 75)]
    #[case(12, 16)]
    fn test_smart_change_in_mean_gain(#[case] start: usize, #[case] stop: usize) {
        let X = testing::array();
        let X_view = X.view();

        assert_eq!(X_view.shape(), &[100, 5]);

        let mut change_in_mean = ChangeInMean::new(&X_view);
        let mut simple_change_in_mean = testing::ChangeInMean::new(&X_view);

        for split in start..stop {
            assert_approx_eq!(
                change_in_mean.gain(start, stop, split),
                simple_change_in_mean.gain(start, stop, split)
            );
        }
    }
}
