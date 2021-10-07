use crate::control::Control;
use crate::gain::Gain;
use crate::model_selection::ModelSelection;
use crate::optimizer::Optimizer;
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use rand;
use std::cell::{Ref, RefCell};

#[allow(non_camel_case_types)]
#[allow(dead_code)]
pub struct kNN<'a, 'b> {
    X: &'a ArrayView2<'b, f64>,
    ordering: RefCell<Option<Array2<usize>>>,
}

impl<'a, 'b> kNN<'a, 'b> {
    #[allow(dead_code)]
    pub fn new(X: &'a ArrayView2<'b, f64>) -> kNN<'a, 'b> {
        kNN {
            X,
            ordering: RefCell::new(Option::None),
        }
    }

    #[allow(dead_code)]
    fn calculate_ordering(&self) -> Array2<usize> {
        let n = self.X.nrows();
        let mut distances = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i >= j {
                    distances[[i, j]] = distances[[j, i]]
                } else {
                    for k in 0..self.X.ncols() {
                        distances[[i, j]] += (self.X[[i, k]] - self.X[[j, k]]).powi(2)
                    }
                }
            }
        }

        // A rather complex ordering = numpy.argsort(distances, 1)
        let mut ordering = Array2::<usize>::default((n, n));
        for (i, mut row) in ordering.axis_iter_mut(Axis(0)).enumerate() {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|a, b| {
                distances[[i, *a]].partial_cmp(&distances[[i, *b]]).unwrap()
            });
            for (j, val) in row.iter_mut().enumerate() {
                *val = order[j]
            }
        }
        ordering
    }

    fn get_ordering(&self) -> Ref<Array2<usize>> {
        if self.ordering.borrow().is_none() {
            self.ordering.replace(Some(self.calculate_ordering()));
        }

        Ref::map(self.ordering.borrow(), |borrow| borrow.as_ref().unwrap())
    }

    fn predictions(&self, start: usize, stop: usize, split: usize) -> Array1<f64> {
        let ordering = self.get_ordering();
        let segment_length = stop - start;
        let k = (segment_length as f64).sqrt().floor();
        let k_usize = k as usize;
        let mut predictions = Array1::<f64>::zeros(segment_length);

        for (i, row) in ordering
            .slice(s![start..stop, ..])
            .axis_iter(Axis(0))
            .enumerate()
        {
            predictions[i] = row // order of neighbors by distance
                .iter()
                .skip(1) // To get LOOCV-like predictions
                .filter(|j| (start <= **j) & (**j < stop)) // segment
                .take(k_usize) // Only look at first k neighbors
                .filter(|j| **j >= split)
                .count() as f64
                / k; // Proportion of neighbors from after split.
        }

        predictions
    }
}

impl<'a, 'b> Gain for kNN<'a, 'b> {
    fn n(&self) -> usize {
        self.X.nrows()
    }

    fn gain(&self, start: usize, stop: usize, split: usize) -> f64 {
        if (split - start <= 1) | (stop - split <= 1) {
            return 0.;
        }

        let predictions = self.predictions(start, stop, split);
        prediction_log_likelihood(predictions, start, stop, split)
    }
}

impl<'a, 'b> Optimizer for kNN<'a, 'b> {}
impl<'a, 'b> ModelSelection for kNN<'a, 'b> {
    fn is_significant(&self, start: usize, stop: usize, split: usize, _: Control) -> bool {
        let mut predictions = self.predictions(start, stop, split);
        println!(
            "start={}, stop={}, split={}, \n predictions={}",
            start, stop, split, predictions
        );
        let prior_00 = ((stop - start - 1) as f64) / ((split - start - 1) as f64);
        let prior_01 = ((stop - start - 1) as f64) / ((split - start) as f64);
        let prior_10 = ((stop - start - 1) as f64) / ((stop - split - 1) as f64);
        let prior_11 = ((stop - start - 1) as f64) / ((stop - split) as f64);

        predictions
            .slice_mut(s![..(split - start)])
            .mapv_inplace(|x| log_eta((1. - x) * prior_00) - log_eta(x * prior_01));
        predictions
            .slice_mut(s![(split - start)..])
            .mapv_inplace(|x| log_eta((1. - x) * prior_10) - log_eta(x * prior_11));

        let n_permutations = 99;

        let mut rng = rand::thread_rng();

        let mut gain = 0.;
        let mut value = 0.;

        for idx in 0..(stop - start) {
            value += predictions[idx];
            if value > gain {
                gain = value;
            }
        }

        let mut p_value: u32 = 0;

        for _ in 0..n_permutations {
            value = 0.;
            for idx in rand::seq::index::sample(&mut rng, stop - start, stop - start) {
                value += predictions[idx];
                if value > gain {
                    p_value += 1;
                    break;
                }
            }
        }
        println!("{}", p_value);
        p_value < 5
    }
}

fn prediction_log_likelihood(
    predictions: Array1<f64>,
    start: usize,
    stop: usize,
    split: usize,
) -> f64 {
    let (left, right) = predictions.slice(s![..]).split_at(Axis(0), split - start);
    let left_correction = ((stop - start - 1) as f64) / ((split - start - 1) as f64);
    let right_correction = ((stop - start - 1) as f64) / ((stop - split - 1) as f64);
    left.mapv(|x| log_eta((1. - x) * left_correction)).sum()
        + right.mapv(|x| log_eta(x * right_correction)).sum()
}

fn log_eta(x: f64) -> f64 {
    // 1e-6 ~ 0.00247, 1 - 1e-6 ~ 0.99752
    (0.00247875217 + 0.99752124782 * x).ln()
}

#[cfg(test)]
mod tests {

    use super::*;
    use assert_approx_eq::*;
    use ndarray::arr1;
    use rstest::*;

    #[test]
    fn test_X_ordering() {
        let X = ndarray::array![[1.], [1.5], [3.], [-0.5]];
        let X_view = X.view();

        let knn = kNN::new(&X_view);
        let ordering = knn.calculate_ordering();
        let expected = ndarray::array![[0, 1, 3, 2], [1, 0, 2, 3], [2, 1, 0, 3], [3, 0, 1, 2]];
        assert_eq!(ordering, expected)
    }

    #[rstest]
    #[case(0, 6, 2, arr1(&[0.5, 0.5, 0., 1., 1., 0.5]))]
    #[case(0, 6, 3, arr1(&[0., 0., 0., 1., 1., 0.5]))]
    #[case(1, 6, 2, arr1(&[1., 0.5, 1., 1., 0.5]))]
    #[case(1, 5, 2, arr1(&[1., 0.5, 0.5, 0.5]))]
    #[case(1, 5, 5, arr1(&[0., 0., 0., 0.]))]
    #[case(2, 2, 2, arr1(&[]))]
    fn test_predictions(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] split: usize,
        #[case] expected: Array1<f64>,
    ) {
        let X = ndarray::array![
            [1., 1.],
            [1.5, 1.],
            [0.5, 1.],
            [3., 3.],
            [4.5, 3.],
            [2.5, 2.5]
        ];
        let X_view = X.view();

        let knn = kNN::new(&X_view);
        let predictions = knn.predictions(start, stop, split);

        assert_eq!(predictions, expected);
    }

    #[rstest]
    #[case(0, 6, arr1(&[0.0, 0.0, -3.3325539228390255, 4.796659545476027, -9.55569673879512, 0.0]))]
    fn test_gain(#[case] start: usize, #[case] stop: usize, #[case] expected: Array1<f64>) {
        // TODO Find out if this makes any sense.
        let X = ndarray::array![
            [1., 1.],
            [1.5, 1.],
            [0.5, 1.],
            [3., 3.],
            [4.5, 3.],
            [2.5, 2.5]
        ];
        let X_view = X.view();

        let knn = kNN::new(&X_view);
        let mut gain = ndarray::Array::from_elem(6, f64::NAN);

        for split_point in start..stop {
            gain[split_point] = knn.gain(start, stop, split_point);
        }
        for idx in start..stop {
            assert_approx_eq!(gain[idx], expected[idx]);
        }
    }
}
