use crate::control::Control;
use crate::gain::Gain;
use crate::model_selection::ModelSelection;
use crate::optimizer::Optimizer;
use crate::utils::log_eta;
use ndarray::{s, stack, Array1, Array2, Axis};

pub trait Classifier {
    fn predict(&self, start: usize, stop: usize, split: usize) -> Array1<f64>;

    fn single_likelihood(
        &self,
        predictions: &Array1<f64>,
        start: usize,
        stop: usize,
        split: usize,
    ) -> f64 {
        if (stop - split <= 1) || (split - start <= 1) {
            return 0.;
        }

        let (left, right) = predictions.slice(s![..]).split_at(Axis(0), split - start);
        let left_correction = ((stop - start - 1) as f64) / ((split - start - 1) as f64);
        let right_correction = ((stop - start - 1) as f64) / ((stop - split - 1) as f64);
        left.mapv(|x| log_eta((1. - x) * left_correction)).sum()
            + right.mapv(|x| log_eta(x * right_correction)).sum()
    }

    fn full_likelihood(
        &self,
        predictions: &Array1<f64>,
        start: usize,
        stop: usize,
        split: usize,
    ) -> Array2<f64> {
        if (stop - split <= 1) || (split - start <= 1) {
            return Array2::zeros((2, stop - start));
        }
        println!(
            "start={}, stop={}, split={}, predictions={}",
            start, stop, split, predictions
        );
        let mut likelihoods = stack(Axis(0), &[predictions.view(), predictions.view()]).unwrap();
        assert!(likelihoods.shape() == [2, stop - start]);

        let prior_00 = ((stop - start - 1) as f64) / ((split - start - 1) as f64);
        let prior_01 = ((stop - start - 1) as f64) / ((split - start) as f64);
        let prior_10 = ((stop - start - 1) as f64) / ((stop - split) as f64);
        let prior_11 = ((stop - start - 1) as f64) / ((stop - split - 1) as f64);

        likelihoods
            .slice_mut(s![0, ..(split - start)])
            .mapv_inplace(|x| log_eta((1. - x) * prior_00));
        likelihoods
            .slice_mut(s![0, (split - start)..])
            .mapv_inplace(|x| log_eta((1. - x) * prior_01));
        likelihoods
            .slice_mut(s![1, ..(split - start)])
            .mapv_inplace(|x| log_eta(x * prior_10));
        likelihoods
            .slice_mut(s![1, (split - start)..])
            .mapv_inplace(|x| log_eta(x * prior_11));

        likelihoods
    }

    fn n(&self) -> usize;
}

pub struct ClassifierGain<T: Classifier> {
    pub classifier: T,
}

impl<T> Gain for ClassifierGain<T>
where
    T: Classifier,
{
    fn gain(&self, start: usize, stop: usize, split: usize) -> f64 {
        let predictions = self.classifier.predict(start, stop, split);
        self.classifier
            .single_likelihood(&predictions, start, stop, split)
    }

    fn gain_approx(
        &self,
        start: usize,
        stop: usize,
        guess: usize,
        _: Vec<usize>,
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
        gain.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);

        gain + likelihoods.slice(s![1, ..]).sum()
    }

    fn n(&self) -> usize {
        self.classifier.n()
    }
}

impl<T> ModelSelection for ClassifierGain<T>
where
    T: Classifier,
{
    fn is_significant(&self, start: usize, stop: usize, split: usize, control: Control) -> bool {
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
        println!("{}", p_value);
        (p_value as f64 / n_permutations as f64) < control.alpha
    }
}

impl<T> Optimizer for ClassifierGain<T> where T: Classifier {}
