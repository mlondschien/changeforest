use crate::control::Control;
use crate::gain::Gain;
use crate::optimizer::OptimizerResult;
use crate::ModelSelectionResult;
use std::cell::{Ref, RefCell};

pub struct ChangeInMean<'a, 'b> {
    X: &'a ndarray::ArrayView2<'b, f64>,
    X_cumsum: RefCell<Option<ndarray::Array2<f64>>>,
    control: &'a Control,
}

impl<'a, 'b> ChangeInMean<'a, 'b> {
    pub fn new(X: &'a ndarray::ArrayView2<'b, f64>, control: &'a Control) -> ChangeInMean<'a, 'b> {
        ChangeInMean {
            X,
            X_cumsum: RefCell::new(Option::None),
            control,
        }
    }

    fn calculate_cumsum(&self) -> ndarray::Array2<f64> {
        let mut X_cumsum = ndarray::Array2::zeros((self.X.nrows() + 1, self.X.ncols()));
        let mut slice = X_cumsum.slice_mut(ndarray::s![1.., ..]);
        slice += &self.X.view();

        X_cumsum.accumulate_axis_inplace(ndarray::Axis(0), |&prev, curr| *curr += prev);
        X_cumsum
    }

    fn get_cumsum(&self) -> Ref<ndarray::Array2<f64>> {
        if self.X_cumsum.borrow().is_none() {
            self.X_cumsum.replace(Some(self.calculate_cumsum()));
        }

        Ref::map(self.X_cumsum.borrow(), |borrow| borrow.as_ref().unwrap())
    }
}

impl<'a, 'b> Gain for ChangeInMean<'a, 'b> {
    fn n(&self) -> usize {
        self.X.nrows()
    }

    fn gain(&self, start: usize, stop: usize, split: usize) -> f64 {
        if (start == split) | (split == stop) {
            return 0.;
        }

        let X_cumsum = self.get_cumsum();

        let s_1 = (split - start) as f64;
        let s_2 = (stop - split) as f64;
        let s = s_1 + s_2;

        let mut result = 0.;
        for idx in 0..self.X.ncols() {
            result += (s_1 * X_cumsum[[stop, idx]] + s_2 * X_cumsum[[start, idx]]
                - s * X_cumsum[[split, idx]])
            .powi(2)
        }
        result / (s * s_1 * s_2)
    }

    fn model_selection(&self, optimizer_result: &OptimizerResult) -> ModelSelectionResult {
        let minimal_gain_to_split = match self.control.minimal_gain_to_split {
            Some(minimal_gain_to_split) => minimal_gain_to_split,
            // log(n) * (d + 1), where (d + 1) is the number of additional parameters through
            // an additional changepoint. See also
            // Yao, Y.-C. (1988). Estimating the number of change-points via Schwarz’ criterion.
            None => (self.X.shape()[0] as f64).ln() * (self.X.shape()[1] as f64 + 1.),
        };

        ModelSelectionResult {
            is_significant: optimizer_result.max_gain > minimal_gain_to_split,
            p_value: None,
        }
    }

    fn control(&self) -> &Control {
        self.control
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::testing;
    use assert_approx_eq::*;
    use rstest::*;

    #[test]
    fn test_X_cumsum() {
        let X = ndarray::array![[1., 0.], [1., 0.], [1., 1.], [1., 1.]];
        let X_view = X.view();
        let control = Control::default();

        let change_in_mean = ChangeInMean::new(&X_view, &control);
        let X_cumsum = change_in_mean.calculate_cumsum();

        let expected = ndarray::array![[0., 0.], [1., 0.], [2., 0.], [3., 1.], [4., 2.]];
        assert_eq!(X_cumsum, expected);
    }

    #[rstest]
    #[case(0, 100)]
    #[case(0, 75)]
    #[case(12, 16)]
    fn test_smart_change_in_mean_gain(#[case] start: usize, #[case] stop: usize) {
        let X = testing::array();
        let X_view = X.view();
        let control = Control::default();

        assert_eq!(X_view.shape(), &[100, 5]);

        let change_in_mean = ChangeInMean::new(&X_view, &control);
        let simple_change_in_mean = testing::ChangeInMean::new(&X_view, &control);

        for split in start..stop {
            assert_approx_eq!(
                change_in_mean.gain(start, stop, split),
                simple_change_in_mean.gain(start, stop, split)
            );
        }
    }
}
