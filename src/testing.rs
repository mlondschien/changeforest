#[cfg(test)]
pub mod testing {
    use super::super::gain;
    use super::super::model_selection::ModelSelection;
    use super::super::optimizer;
    use ndarray::{s, Array, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    pub struct ChangeInMean<'a> {
        X: &'a ndarray::Array2<f64>,
    }

    impl<'a> ChangeInMean<'a> {
        #[allow(dead_code)]
        pub fn new(X: &'a ndarray::Array2<f64>) -> ChangeInMean<'a> {
            ChangeInMean { X: &X }
        }
    }

    impl<'a> gain::Gain for ChangeInMean<'a> {
        fn n(&self) -> usize {
            self.X.nrows()
        }

        fn loss(&self, start: usize, stop: usize) -> f64 {
            if start == stop {
                return 0.;
            };

            let n_total = self.X.nrows() as f64;
            let n_slice = (stop - start) as f64;

            let slice = &self.X.slice(ndarray::s![start..stop, ..]);

            // For 1D, the change in mean loss is equal to
            // 1 / n_total * [sum_i x_i**2 - 1/n_slice (sum_i x_i)**2]
            // For 2D, the change in mean loss is just the sum of losses for each dimension.
            let loss = slice.mapv(|a| a.powi(2)).sum()
                - slice.sum_axis(ndarray::Axis(0)).mapv(|a| a.powi(2)).sum() / n_slice;

            loss / n_total
        }
    }

    impl<'a> optimizer::Optimizer for ChangeInMean<'a> {}

    impl<'a> ModelSelection for ChangeInMean<'a> {}

    pub fn array() -> Array2<f64> {
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);

        let mut X = Array::zeros((100, 5)); //

        X.slice_mut(s![0..25, 0]).fill(2.);
        X.slice_mut(s![40..80, 0]).fill(1.);
        X.slice_mut(s![0..40, 1]).fill(-2.);
        X.slice_mut(s![40..100, 1]).fill(-3.);

        X + Array::random_using((100, 5), Uniform::new(0., 1.), &mut rng)
    }
}
