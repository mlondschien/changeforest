use crate::Classifier;
use ndarray::{s, Array1, ArrayView2};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};

#[allow(non_camel_case_types)]
pub struct RandomForest<'a, 'b> {
    X: &'a ArrayView2<'b, f64>,
}

impl<'a, 'b> RandomForest<'a, 'b> {
    #[allow(dead_code)]
    pub fn new(X: &'a ArrayView2<'b, f64>) -> RandomForest<'a, 'b> {
        RandomForest { X }
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

        RandomForestRegressor::fit(
            &X_slice,
            &y,
            RandomForestRegressorParameters {
                max_depth: Some(4),
                min_samples_leaf: 1,
                min_samples_split: 2,
                n_trees: 100,
                m: Option::None,
                keep_samples: true,
            },
        )
        .and_then(|rf| rf.predict_oob(&X_slice))
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gain::ClassifierGain;
    use crate::optimizer::{Optimizer, TwoStepSearch};
    use crate::testing;
    use crate::Control;
    use rstest::*;

    // TODO: Impossible without seed in RandomForestRegressorParameters
    // #[rstest]
    // #[case(0, 6, 2, arr1(&[0.66, 0.37, 0.04, 0.89, 0.94, 0.88]))]
    // fn test_predictions(
    //     #[case] start: usize,
    //     #[case] stop: usize,
    //     #[case] split: usize,
    //     #[case] expected: Array1<f64>,
    // ) {
    //     let X = ndarray::array![
    //         [1., 1.],
    //         [1.5, 1.],
    //         [0.5, 1.],
    //         [3., 3.],
    //         [4.5, 3.],
    //         [2.5, 2.5]
    //     ];
    //     let X_view = X.view();

    //     let knn = RandomForest::new(&X_view);
    //     let predictions = knn.predict(start, stop, split);

    //     for (p, e) in predictions.iter().zip(expected) {
    //         assert_approx_eq!(p, e, 1e-2);
    //     }
    // }

    #[rstest]
    #[case(0, 100, 40)]
    #[case(40, 90, 80)]
    fn test_two_step_search(#[case] start: usize, #[case] stop: usize, #[case] expected: usize) {
        let X = testing::array();
        let X_view = X.view();

        let classifier = RandomForest::new(&X_view);
        let gain = ClassifierGain { classifier };
        let control = Control::default().with_minimal_relative_segment_length(0.01);
        let optimizer = TwoStepSearch {
            gain,
            control: &control,
        };

        assert_eq!(
            expected,
            optimizer.find_best_split(start, stop).unwrap().best_split
        );
    }
}
