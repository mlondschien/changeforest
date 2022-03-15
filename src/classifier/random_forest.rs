use crate::{Classifier, Control};
use biosphere::RandomForest as BioForest;
use ndarray::{s, Array1, ArrayView2};

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
        let y_slice = y.slice(s![..]);

        let X_slice = self.X.slice(s![start..stop, ..]);
        let parameters = self.control().random_forest_parameters.clone();

        let mut forest = BioForest::new(parameters);
        let mut predictions = forest.fit_predict_oob(&X_slice, &y_slice);

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
    use csv::ReaderBuilder;
    use ndarray::Array2;
    use ndarray_csv::Array2Reader;
    use rstest::*;
    use std::fs::File;

    #[rstest]
    #[case(0, 50, 100)]
    #[case(0, 100, 150)]
    #[case(50, 100, 150)]
    #[case(0, 50, 150)]
    fn test_predictions(#[case] start: usize, #[case] split: usize, #[case] stop: usize) {
        let file = File::open("testdata/iris.csv").unwrap();
        let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
        let X: Array2<f64> = reader.deserialize_array2((150, 4)).unwrap();
        let X_view = X.view();

        let control = Control::default();

        let rf = RandomForest::new(&X_view, &control);
        let predictions = rf.predict(start, stop, split);

        let mut y = Array1::<f64>::zeros(stop - start);
        y.slice_mut(s![(split - start)..]).fill(1.);

        let mse = (y - predictions).mapv(|x| x.powi(2)).mean().unwrap();
        assert!(mse < 0.06, "mse = {}", mse);
    }

    #[rstest]
    #[case(0, 100, 25)]
    #[case(10, 90, 40)]
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
