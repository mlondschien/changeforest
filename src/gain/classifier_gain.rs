use crate::Classifier;
use crate::Gain;
use ndarray::{s, Array1, Axis};

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
    ) -> ndarray::Array1<f64> {
        let predictions = self.classifier.predict(start, stop, guess);
        let likelihoods = self
            .classifier
            .full_likelihood(&predictions, start, stop, guess);

        let mut gain = Array1::<f64>::zeros(stop - start);
        // Move everything one to the right.
        gain.slice_mut(s![1..]).assign(
            &(&likelihoods.slice(s![0, ..(stop - start - 1)])
                - &likelihoods.slice(s![1, ..(stop - start - 1)])),
        );
        gain.slice_mut(s![start..stop])
            .accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

        gain + likelihoods.slice(s![1, ..]).sum()
    }

    fn is_significant(&self, start: usize, stop: usize, split: usize, _: f64) -> bool {
        let predictions = self.classifier.predict(start, stop, split);
        let full_likelihood = self
            .classifier
            .full_likelihood(&predictions, start, stop, split);
        let delta = &full_likelihood.slice(s![0, ..]) - &full_likelihood.slice(s![1, ..]);
        let n_permutations = 99;

        let mut rng = rand::thread_rng();

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
        (p_value as f64 / (n_permutations + 1) as f64) < 0.05
    }
}
