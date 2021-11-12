use crate::{Classifier, Control};
use ndarray::{s, Array1, ArrayView2};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};

pub struct RandomForest<'a, 'b> {
    X: &'a ArrayView2<'b, f64>,
    control: &'a Control,
}

impl<'a, 'b> RandomForest<'a, 'b> {
    pub fn new(X: &'a ArrayView2<'b, f64>, control: &'a Control) -> RandomForest<'a, 'b> {
        RandomForest { X, control }
    }
}

impl<'a, 'b> Classifier for RandomForest<'a, 'b> {
    fn n(&self) -> usize {
        self.X.nrows()
    }

    fn predict(&self, start: usize, stop: usize, split: usize) -> Array1<f64> {
        let mut y = Array1::<f64>::zeros(stop - start);
        y.slice_mut(s![(split - start)..]).fill(1.);

        let X_slice = self.X.slice(s![start..stop, ..]).to_owned();

        let mut predictions = RandomForestRegressor::fit(
            &X_slice,
            &y,
            RandomForestRegressorParameters {
                max_depth: Some(4),
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: self.control.random_forest_ntrees,
                m: Option::None,
                keep_samples: true,
                seed: self.control.seed,
            },
        )
        .and_then(|rf| rf.predict_oob(&X_slice))
        .unwrap();

        // For a very small n_trees, the predictions may be NaN. In this case use the
        // prior. Note that we need to adjust by -1 because the predictions are oob.
        predictions
            .slice_mut(s![0..(split - start)])
            .map_inplace(|x| {
                if x.is_nan() {
                    *x = (stop - split) as f64 / (stop - start - 1) as f64
                }
            });
        predictions
            .slice_mut(s![(split - start)..])
            .map_inplace(|x| {
                if x.is_nan() {
                    *x = (stop - split - 1) as f64 / (stop - start - 1) as f64
                }
            });
        predictions
    }

    fn control(&self) -> &Control {
        self.control
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gain::ClassifierGain;
    use crate::optimizer::{Optimizer, TwoStepSearch};
    use crate::testing;
    use crate::Control;
    use assert_approx_eq::*;
    use ndarray::arr1;
    use rstest::*;

    #[rstest]
    #[case(0, 6, 2, 0, 100, arr1(&[0.72, 0.32, 0.057, 0.89, 0.95, 0.91]))]
    // What a difference a seed can make.
    #[case(0, 6, 2, 87, 100, arr1(&[0.70, 0.44, 0.0, 1.0, 1.0, 0.95]))]
    #[case(0, 6, 4, 0, 100, arr1(&[0.09, 0.071, 0.08, 0.97, 0.29, 0.18]))]
    #[case(0, 6, 2, 0, 10, arr1(&[0.8, 0.125, 0., 1., 1., 1.]))]
    fn test_predictions(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] split: usize,
        #[case] seed: u64,
        #[case] random_forest_ntrees: usize,
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
        let control = Control::default()
            .with_seed(seed)
            .with_random_forest_ntrees(random_forest_ntrees);

        let rf = RandomForest::new(&X_view, &control);
        let predictions = rf.predict(start, stop, split);

        for (p, e) in predictions.iter().zip(expected) {
            assert_approx_eq!(p, e, 1e-2);
        }
    }

    #[rstest]
    #[case(0, 100, 40)]
    #[case(40, 90, 80)]
    fn test_two_step_search(#[case] start: usize, #[case] stop: usize, #[case] expected: usize) {
        let X = testing::array();
        let X_view = X.view();
        let control = Control::default().with_minimal_relative_segment_length(0.01);
        let classifier = RandomForest::new(&X_view, &control);
        let gain = ClassifierGain { classifier };
        let optimizer = TwoStepSearch { gain };

        assert_eq!(
            expected,
            optimizer.find_best_split(start, stop).unwrap().best_split
        );
    }
}
