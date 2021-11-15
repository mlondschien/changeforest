use crate::control::Control;
use crate::gain::{ApproxGain, ApproxGainResult, Gain, GainResult};
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

    /// Return classifier-likelihood based gain when splitting segment `[start, stop)`
    /// at `split`.
    fn gain(&self, start: usize, stop: usize, split: usize) -> f64 {
        let predictions = self.classifier.predict(start, stop, split);
        self.classifier
            .single_likelihood(&predictions, start, stop, split)
    }

    fn model_selection(&self, _: f64, gain_result: &GainResult) -> ModelSelectionResult {
        let likelihoods: &Array2<f64>;
        let start: usize;
        let stop: usize;

        if let GainResult::ApproxGainResult(result) = gain_result {
            likelihoods = &result.likelihoods;
            start = result.start;
            stop = result.stop;
        } else {
            panic!();
        }

        let delta = &likelihoods.slice(s![0, ..]) - &likelihoods.slice(s![1, ..]);
        let n_permutations = 99;

        let mut rng = StdRng::seed_from_u64(self.classifier.control().seed);

        let mut max_gain = 0.;
        let mut value = 0.;

        for idx in 0..(stop - start) {
            value += delta[idx];
            if value > max_gain {
                max_gain = value;
            }
        }

        let mut p_value: u32 = 1;

        for _ in 0..n_permutations {
            value = 0.;
            for idx in rand::seq::index::sample(&mut rng, stop - start, stop - start) {
                value += delta[idx];
                if value > max_gain {
                    p_value += 1;
                    break;
                }
            }
        }

        let p_value = p_value as f64 / (n_permutations + 1) as f64;
        let is_significant = p_value < self.control().model_selection_alpha;
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
    /// Return an approximation of the classifier-likelihood based gain when splitting
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
