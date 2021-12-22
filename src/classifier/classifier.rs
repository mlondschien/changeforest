use crate::utils::log_eta;
use crate::Control;
use ndarray::{s, stack, Array1, Array2, Axis};

pub trait Classifier {
    fn n(&self) -> usize;

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

    fn control(&self) -> &Control;
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::testing::TrivialClassifier;
    use assert_approx_eq::*;

    #[test]
    fn test_single_likelihood() {
        let control = Control::default();
        let classifier = TrivialClassifier {
            n: 10,
            control: &control,
        };
        let predictions = classifier.predict(0, 10, 5);
        assert_approx_eq!(
            classifier.single_likelihood(&predictions, 0, 10, 5),
            0.809552182
        );
    }

    #[test]
    fn test_full_likelihood() {
        let control = Control::default();
        let classifier = TrivialClassifier {
            n: 10,
            control: &control,
        };
        let predictions = classifier.predict(0, 10, 5);
        let mut expected = Array2::<f64>::zeros((2, 10));
        expected[[0, 0]] = 0.8095521826214339; // TODO: Why not 6.0
        expected[[1, 0]] = -6.0;

        assert_eq!(classifier.full_likelihood(&predictions, 0, 10, 5), expected);
    }
}
