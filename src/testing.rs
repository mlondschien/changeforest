use crate::classifier::Classifier;
use crate::gain::{gain_from_likelihoods, ApproxGain, ApproxGainResult, Gain};
use crate::optimizer::OptimizerResult;
use crate::{Control, ModelSelectionResult, Optimizer};
use ndarray::{s, stack, Array, Array1, Array2, ArrayView2, Axis};
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct ChangeInMean<'a> {
    X: &'a ndarray::ArrayView2<'a, f64>,
    control: &'a Control,
}

impl<'a> ChangeInMean<'a> {
    pub fn new(X: &'a ArrayView2<'a, f64>, control: &'a Control) -> ChangeInMean<'a> {
        ChangeInMean { X, control }
    }
}

impl<'a> Gain for ChangeInMean<'a> {
    fn n(&self) -> usize {
        self.X.nrows()
    }

    fn loss(&self, start: usize, stop: usize) -> f64 {
        if start == stop {
            return 0.;
        };

        let n_slice = (stop - start) as f64;

        let slice = &self.X.slice(s![start..stop, ..]);

        // For 1D, the change in mean loss is equal to
        // 1 / n_total * [sum_i x_i**2 - 1/n_slice (sum_i x_i)**2]
        // For 2D, the change in mean loss is just the sum of losses for each dimension.

        slice.mapv(|a| a.powi(2)).sum()
            - slice.sum_axis(Axis(0)).mapv(|a| a.powi(2)).sum() / n_slice
    }

    fn model_selection(&self, optimizer_result: &OptimizerResult) -> ModelSelectionResult {
        ModelSelectionResult {
            is_significant: optimizer_result.max_gain
                > self.control.minimal_gain_to_split.unwrap_or(0.1) * (self.n() as f64),
            p_value: None,
        }
    }

    fn control(&self) -> &Control {
        self.control
    }
}

impl<'a> ApproxGain for ChangeInMean<'a> {
    fn gain_approx(
        &self,
        start: usize,
        stop: usize,
        guess: usize,
        _: &[usize],
    ) -> ApproxGainResult {
        let slice = self.X.slice(s![start..stop, ..]);
        let mut left_fit = self.X.slice(s![start..guess, ..]).sum_axis(Axis(0));
        let mut right_fit = self.X.slice(s![guess..stop, ..]).sum_axis(Axis(0));

        let overall_fit = (&left_fit + &right_fit) / ((stop - start) as f64);
        left_fit.mapv_inplace(|x| x / (guess - start) as f64);
        right_fit.mapv_inplace(|x| x / (stop - guess) as f64);

        let likelihoods = stack(
            Axis(0),
            &[
                (slice.dot(&(&left_fit - &overall_fit)).mapv(|x| x * 2.0)
                    + (&overall_fit.map(|x| x.powi(2)) - &left_fit.map(|x| x.powi(2)))
                        .sum_axis(Axis(0)))
                .view(),
                (slice.dot(&(&right_fit - &overall_fit)).mapv(|x| x * 2.0)
                    + (&overall_fit.map(|x| x.powi(2)) - &right_fit.map(|x| x.powi(2)))
                        .sum_axis(Axis(0)))
                .view(),
            ],
        )
        .unwrap();
        // TODO Change this to [n, 2];
        assert!(likelihoods.shape() == [2, stop - start]);

        let gain = gain_from_likelihoods(&likelihoods);
        let predictions = likelihoods.map_axis(Axis(0), |x| {
            x[1].exp() * (stop - guess) as f64 / (stop - start) as f64
        });

        ApproxGainResult {
            start,
            stop,
            guess,
            gain,
            best_split: None,
            max_gain: None,
            predictions,
            likelihoods,
        }
    }
}

pub struct TrivialOptimizer<'a> {
    pub control: &'a Control,
}

impl<'a> Optimizer for TrivialOptimizer<'a> {
    fn n(&self) -> usize {
        100
    }

    fn find_best_split(&self, start: usize, stop: usize) -> Result<OptimizerResult, &'static str> {
        Ok(OptimizerResult {
            start,
            stop,
            best_split: (3 * start + stop) / 4,
            max_gain: ((stop - start) * (start + 10)) as f64,
            gain_results: vec![],
        })
    }

    fn model_selection(&self, optimizer_result: &OptimizerResult) -> ModelSelectionResult {
        ModelSelectionResult {
            is_significant: optimizer_result.stop <= 50,
            p_value: None,
        }
    }

    fn control(&self) -> &Control {
        self.control
    }
}

pub struct TrivialClassifier<'a> {
    pub n: usize,
    pub control: &'a Control,
}

impl<'a> Classifier for TrivialClassifier<'a> {
    fn n(&self) -> usize {
        self.n
    }

    fn predict(&self, start: usize, stop: usize, split: usize) -> Array1<f64> {
        let mut X = Array::zeros(stop - start);
        X.slice_mut(s![0..(split - start)])
            .fill((stop - split) as f64 / (stop - start - 1) as f64);
        X.slice_mut(s![(split - start)..])
            .fill((stop - split - 1) as f64 / (stop - start - 1) as f64);
        X[[0]] = 0.;
        X
    }

    fn control(&self) -> &Control {
        self.control
    }
}

/// Classifier that predicts uniformly distributed values.
pub struct RandomClassifier<'a> {
    pub n: usize,
    pub control: &'a Control,
    pub seed: u64,
}

impl<'a> Classifier for RandomClassifier<'a> {
    fn n(&self) -> usize {
        self.n
    }

    fn predict(&self, start: usize, stop: usize, guess: usize) -> Array1<f64> {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut predictions = Array1::zeros(stop - start);
        let left = Array1::random_using(
            guess - start,
            Normal::new((stop - guess) as f64 / (stop - start - 1) as f64, 0.1).unwrap(),
            &mut rng,
        );
        let right = Array1::random_using(
            stop - guess,
            Normal::new((stop - guess) as f64 / (stop - start - 1) as f64, 0.1).unwrap(),
            &mut rng,
        );
        predictions.slice_mut(s![..(guess - start)]).assign(&left);
        predictions.slice_mut(s![(guess - start)..]).assign(&right);
        predictions.mapv_inplace(|x| f64::min(f64::max(0., x), 1.));
        predictions
    }

    fn control(&self) -> &Control {
        self.control
    }
}

pub fn array() -> Array2<f64> {
    let seed = 7;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut X = Array::zeros((100, 5)); //

    X.slice_mut(s![0..25, 0]).fill(2.);
    X.slice_mut(s![40..80, 0]).fill(1.);
    X.slice_mut(s![0..40, 1]).fill(-2.);
    X.slice_mut(s![25..40, 2]).fill(3.);
    X.slice_mut(s![25..80, 1]).fill(-2.);

    X + Array::random_using((100, 5), Uniform::new(0., 1.), &mut rng)
}

#[cfg(test)]
mod tests {

    use super::*;
    use assert_approx_eq::*;
    use ndarray::array;
    use rstest::*;

    #[rstest]
    #[case(0, 6, 0.25 * 6.)]
    #[case(0, 3, 0.)]
    #[case(3, 6, 0.)]
    #[case(2, 4, 0.25 * 2.)]
    fn test_loss(#[case] start: usize, #[case] stop: usize, #[case] expected: f64) {
        let X = ndarray::array![[0.], [0.], [0.], [1.], [1.], [1.]];
        let X_view = X.view();
        let control = Control::default();

        let change_in_mean = ChangeInMean::new(&X_view, &control);
        assert_eq!(change_in_mean.loss(start, stop), expected)
    }

    #[rstest]
    #[case(0, 6, 3, array![-1.5, -0.5, 0.5, 1.5, 0.5, -0.5])]
    #[case(1, 5, 3, array![-1., 0., 1., 0.])]
    fn test_approx_gain(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] guess: usize,
        #[case] expected_gain: Array1<f64>,
    ) {
        let X = ndarray::array![[0.], [0.], [0.], [1.], [1.], [1.]];
        let X_view = X.view();
        let control = Control::default();

        let change_in_mean = ChangeInMean::new(&X_view, &control);
        let split_points: Vec<usize> = (start..stop).collect();

        let approx_gain_result = change_in_mean.gain_approx(start, stop, guess, &split_points);

        assert!(approx_gain_result.gain.abs_diff_eq(&expected_gain, 1e-8));
        assert_eq!(
            approx_gain_result.gain[guess - start],
            change_in_mean.gain(start, stop, guess)
        );
    }

    #[rstest]
    #[case(0, 100)]
    #[case(0, 99)]
    #[case(12, 83)]
    fn test_compare_approx_to_normal_gain(#[case] start: usize, #[case] stop: usize) {
        let X = array();
        let X_view = X.view();
        let control = Control::default();

        let change_in_mean = ChangeInMean::new(&X_view, &control);
        let split_points: Vec<usize> = (start..stop).collect();

        for guess in start..stop {
            assert_approx_eq!(
                change_in_mean
                    .gain_approx(start, stop, guess, &split_points)
                    .gain[guess - start],
                change_in_mean.gain(start, stop, guess)
            );
        }
    }
}
