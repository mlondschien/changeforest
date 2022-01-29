use crate::{Classifier, Control};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use std::cell::{Ref, RefCell};

#[allow(non_camel_case_types)]
pub struct kNN<'a, 'b> {
    X: &'a ArrayView2<'b, f64>,
    ordering: RefCell<Option<Array2<usize>>>,
    control: &'a Control,
}

impl<'a, 'b> kNN<'a, 'b> {
    pub fn new(X: &'a ArrayView2<'b, f64>, control: &'a Control) -> kNN<'a, 'b> {
        kNN {
            X,
            ordering: RefCell::new(Option::None),
            control,
        }
    }

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
}

impl<'a, 'b> Classifier for kNN<'a, 'b> {
    fn n(&self) -> usize {
        self.X.nrows()
    }

    fn predict(&self, start: usize, stop: usize, split: usize) -> Array1<f64> {
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

    fn control(&self) -> &Control {
        self.control
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gain::{ApproxGain, ClassifierGain, Gain};
    use crate::optimizer::{Optimizer, TwoStepSearch};
    use crate::testing;
    use assert_approx_eq::*;
    use ndarray::arr1;
    use rstest::*;

    #[test]
    fn test_X_ordering() {
        let X = ndarray::array![[1.], [1.5], [3.], [-0.5]];
        let X_view = X.view();
        let control = Control::default();

        let knn = kNN::new(&X_view, &control);
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
        let control = Control::default();

        let knn = kNN::new(&X_view, &control);
        let predictions = knn.predict(start, stop, split);

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
        let control = Control::default();

        let knn = kNN::new(&X_view, &control);
        let knn_gain = ClassifierGain { classifier: knn };

        let split_points: Vec<usize> = (start..stop).collect();
        for split_point in start..stop {
            assert_approx_eq!(
                expected[split_point - start],
                knn_gain.gain(start, stop, split_point)
            );
            assert_approx_eq!(
                expected[split_point - start],
                knn_gain
                    .gain_approx(start, stop, split_point, &split_points)
                    .gain[split_point - start]
            )
        }
    }

    #[rstest]
    #[case(0, 100, 40)]
    fn test_two_step_search(#[case] start: usize, #[case] stop: usize, #[case] expected: usize) {
        let X = testing::array();
        let X_view = X.view();
        let control = Control::default().with_minimal_relative_segment_length(0.01);

        let classifier = kNN::new(&X_view, &control);
        let gain = ClassifierGain { classifier };
        let optimizer = TwoStepSearch { gain };

        assert_eq!(
            expected,
            optimizer.find_best_split(start, stop).unwrap().best_split
        );
    }
}
