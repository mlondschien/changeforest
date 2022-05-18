use crate::control::Control;
use crate::gain::{ApproxGainResult, FullGainResult};
use crate::optimizer::OptimizerResult;
use crate::ModelSelectionResult;

pub trait Gain {
    /// Total number of observations.
    fn n(&self) -> usize;

    #[allow(unused_variables)]
    /// Loss of segment `[start, stop)`.
    ///
    /// This is typically a parametric loss, i.e. minimal negative log-likelihood. Needs
    /// not be normalized the by segment length.
    fn loss(&self, start: usize, stop: usize) -> f64 {
        panic!("Not implemented.");
    }

    /// Gain when splitting segment [start, stop) at `split`.
    fn gain(&self, start: usize, stop: usize, split: usize) -> f64 {
        self.loss(start, stop) - self.loss(start, split) - self.loss(split, stop)
    }

    /// Gain when splitting segment `[start, stop)` at points in `split_candidates`.
    ///
    /// Returns an `ndarray::Array1` of length `stop - start`. Entries without
    /// corresponding entry in `split_candidates` are `f64::NAN`.
    fn gain_full(&self, start: usize, stop: usize, split_candidates: &[usize]) -> FullGainResult {
        let mut gain = ndarray::Array::from_elem(stop - start, f64::NAN);

        for split_point in split_candidates {
            gain[split_point - start] = self.gain(start, stop, *split_point);
        }

        FullGainResult {
            start,
            stop,
            gain,
            max_gain: None,
            best_split: None,
        }
    }

    /// Does a certain split corresponds to a true change point?
    fn model_selection(&self, optimizer_result: &OptimizerResult) -> ModelSelectionResult;

    /// Hyperparameters.
    fn control(&self) -> &Control;
}

pub trait ApproxGain {
    #[allow(unused_variables)]
    /// An approximation of the gain when splitting segment `[start, stop)` at points in `split_candidates`.
    ///
    /// Returns an `ndarray::Array1` of length `stop - start`. Entries without
    /// corresponding entry in `split_candidates` are `f64::NAN`.
    ///
    /// This can be useful when combining classifier-based gains and the two-step-search
    /// optimizer.
    fn gain_approx(
        &self,
        start: usize,
        stop: usize,
        guess: usize,
        split_points: &[usize],
    ) -> ApproxGainResult;
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::testing;
    use assert_approx_eq::*;
    use rstest::*;

    #[rstest]
    #[case(0, 4, 4. * 0.25)]
    #[case(0, 2, 0.)]
    #[case(0, 3, 4. / 6.)]
    #[case(1, 4, 4. / 6.)]
    #[case(1, 3, 4. * 0.125)]
    #[case(3, 3, 0.)]
    fn test_change_in_mean_loss(#[case] start: usize, #[case] stop: usize, #[case] expected: f64) {
        let X = ndarray::array![[0., 0.], [0., 0.], [0., 1.], [0., 1.]];
        let X_view = X.view();

        let control = Control::default();
        assert_eq!(X.shape(), &[4, 2]);

        let change_in_mean = testing::ChangeInMean::new(&X_view, &control);
        assert_approx_eq!(change_in_mean.loss(start, stop), expected);
    }

    #[rstest]
    #[case(0, 4, 2, 1.)]
    #[case(0, 4, 0, 0.)]
    #[case(0, 4, 1, 6. / 18.)]
    #[case(0, 4, 3, 6. / 18.)]
    #[case(0, 3, 2, 6. / 9.)]
    #[case(0, 3, 1, 6. / 36.)]
    #[case(0, 6, 0, 0.)]
    #[case(0, 6, 1, 6. / 90.)]
    #[case(0, 6, 2, 6. / 36.)]
    #[case(0, 6, 3, 6. / 18.)]
    #[case(0, 6, 4, 5. * 6. / 18.)]
    #[case(0, 6, 5, 6. / 90.)]
    fn test_change_in_mean_gain(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] split: usize,
        #[case] expected: f64,
    ) {
        let X = ndarray::array![[1., 0.], [1., 0.], [1., 1.], [1., 1.], [0., -1.], [1., 0.]];
        let X_view = X.view();
        assert_eq!(X_view.shape(), &[6, 2]);

        let control = Control::default();
        let change_in_mean = testing::ChangeInMean::new(&X_view, &control);
        assert_approx_eq!(change_in_mean.gain(start, stop, split), expected);
        assert_approx_eq!(
            change_in_mean.gain_full(start, stop, &[split]).gain[split - start],
            expected
        );
    }
}
