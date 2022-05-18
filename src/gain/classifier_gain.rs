use crate::control::Control;
use crate::gain::{ApproxGain, ApproxGainResult, Gain, GainResult};
use crate::optimizer::OptimizerResult;
use crate::Classifier;
use crate::ModelSelectionResult;
use ndarray::{s, Array1, Array2, Axis};
use rand::{rngs::StdRng, SeedableRng};

pub struct ClassifierGain<T: Classifier> {
    pub classifier: T,
}

impl<T> Gain for ClassifierGain<T>
where
    T: Classifier,
{
    /// Total number of observations.
    fn n(&self) -> usize {
        self.classifier.n()
    }

    /// Return classifier log-likelihood ratio when splitting segment `[start, stop)`
    /// at `split`.
    fn gain(&self, start: usize, stop: usize, split: usize) -> f64 {
        let predictions = self.classifier.predict(start, stop, split);
        self.classifier
            .single_likelihood(&predictions, start, stop, split)
    }

    /// Perform a permutation test.
    ///
    /// We test whether the maximum observed gain from the first step in the
    /// `TwoStepSearch` optimizer is significant. Using the maximum gain from the first
    /// step instead allows us to do a proper permutation test with control of type I
    /// error without fitting additional classifiers.
    ///
    /// In the first step of the `TwoStepSearch` optimizer, three gain curves and
    /// corresponding maximal gains are computed. The maximum gain of the first step
    /// of `TwoStepSearch` is the maximum of these three gains, which are available as
    /// the first three elements of `optimizer_result.gain_results`.
    ///
    /// For each permutation, we shuffle the predictions (and thus the likelihoods) of
    /// each of the three initial classifier fits (using the same permutation), and
    /// compute the maximum of the three resulting maximal gains. We count the number
    /// of permutations where the resulting maximal gain was larger than the observed
    /// maximal gain to compute a p-value.
    fn model_selection(&self, optimizer_result: &OptimizerResult) -> ModelSelectionResult {
        let mut rng = StdRng::seed_from_u64(self.control().seed);

        let mut max_gain = -f64::INFINITY;
        let mut deltas: Vec<Array1<f64>> = Vec::with_capacity(3);
        let mut likelihood_0: Vec<f64> = Vec::with_capacity(3);

        for gain_result in optimizer_result.gain_results.split_last().unwrap().1.iter() {
            let result = match gain_result {
                GainResult::ApproxGainResult(result) => result,
                _ => panic!("Not an ApproxGainResult"),
            };

            deltas
                .push(&result.likelihoods.slice(s![0, ..]) - &result.likelihoods.slice(s![1, ..]));
            likelihood_0.push(result.likelihoods.slice(s![1, ..]).sum());

            if result.max_gain.unwrap() > max_gain {
                max_gain = result.max_gain.unwrap();
            }
        }

        let mut p_value: u32 = 1;
        let segment_length = optimizer_result.stop - optimizer_result.start;

        // ceil(delta * n)
        let minimal_segment_length =
            (self.control().minimal_relative_segment_length * (self.n() as f64)).ceil() as usize;

        for _ in 0..self.control().model_selection_n_permutations {
            let mut values = likelihood_0.clone();
            let permutation = rand::seq::index::sample(&mut rng, segment_length, segment_length);

            for idx in permutation.iter().take(minimal_segment_length - 1) {
                for jdx in 0..deltas.len() {
                    values[jdx] += deltas[jdx][idx];
                }
            }

            // Test if for any jdx=1,2,3 the gain (likelihood_0[jdx] + cumsum(deltas[jdx]))
            // is greater than max_gain. This is the statistic we are comparing against.
            'outer: for idx in permutation
                .iter()
                .skip(minimal_segment_length - 1)
                .take(segment_length - 2 * minimal_segment_length + 1)
            {
                for jdx in 0..deltas.len() {
                    values[jdx] += deltas[jdx][idx];
                    if values[jdx] >= max_gain {
                        p_value += 1;
                        // break both loops. We only need to check if the maximum of the
                        // maximal gain after permutation is ever greater than the
                        // original max_gain (without permutation).
                        break 'outer;
                    }
                }
            }
        }

        // Up to here p_value is # of permutations for which the max_gain is higher than
        // the non-permuted max_gain. From this create a true p_value.
        let p_value = p_value as f64 / (self.control().model_selection_n_permutations + 1) as f64;
        let is_significant = p_value <= self.control().model_selection_alpha;

        ModelSelectionResult {
            is_significant,
            p_value: Some(p_value),
        }
    }

    fn control(&self) -> &Control {
        self.classifier.control()
    }
}

impl<T> ApproxGain for ClassifierGain<T>
where
    T: Classifier,
{
    /// Return an approximation of the classifier log- likelihood ratio when splitting
    /// segment `[start, stop)` for each split in `split_candidates`.
    ///
    /// A single fit is generated with a split at `guess`.
    fn gain_approx(
        &self,
        start: usize,
        stop: usize,
        guess: usize,
        _: &[usize],
    ) -> ApproxGainResult {
        let predictions = self.classifier.predict(start, stop, guess);
        let likelihoods = self
            .classifier
            .full_likelihood(&predictions, start, stop, guess);

        let gain = gain_from_likelihoods(&likelihoods);

        ApproxGainResult {
            start,
            stop,
            guess,
            gain,
            best_split: None,
            max_gain: None,
            likelihoods,
            predictions,
        }
    }
}

pub fn gain_from_likelihoods(likelihoods: &Array2<f64>) -> Array1<f64> {
    let n = likelihoods.shape()[1];
    let mut gain = Array1::<f64>::zeros(n);
    // Move everything one to the right.
    gain.slice_mut(s![1..])
        .assign(&(&likelihoods.slice(s![0, ..(n - 1)]) - &likelihoods.slice(s![1, ..(n - 1)])));
    gain.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

    gain + likelihoods.slice(s![1, ..]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{Optimizer, TwoStepSearch};
    use crate::testing::RandomClassifier;

    #[test]
    fn test_gain_from_likelihoods() {
        let likelihoods = ndarray::array![
            [1., -1.],
            [1., -1.],
            [0.5, -1.5],
            [-2., 0.],
            [-1., 1.],
            [-1., 1.]
        ]
        .reversed_axes();
        let gain = gain_from_likelihoods(&likelihoods);
        let expected = ndarray::array![-1.5, 0.5, 2.5, 4.5, 2.5, 0.5];
        assert_eq!(gain, expected);
    }

    #[test]
    fn test_model_selection() {
        let n = 200;

        let mut p_values = Vec::<f64>::new();

        for seed in 0..100 {
            let control = Control::default();
            let classifier = RandomClassifier {
                n,
                control: &control,
                seed,
            };
            let gain = ClassifierGain { classifier };
            let optimizer = TwoStepSearch { gain };

            let optimizer_result = optimizer.find_best_split(0, n).unwrap();

            let model_selection = optimizer.model_selection(&optimizer_result);
            p_values.push(model_selection.p_value.unwrap());
        }
        let p_value = p_values.into_iter().filter(|x| *x < 0.05).count() as f64 / n as f64;
        assert!(p_value >= 0.03);
        assert!(p_value <= 0.07);
    }
}
