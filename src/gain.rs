use ndarray;

pub trait Segment {
    #[allow(unused_variables)]
    fn loss(&self, start: usize, stop: usize) -> f64 {
        panic!("Not implemented.");
    }

    fn gain(&mut self, start: usize, stop: usize, split: usize) -> f64 {
        self.loss(start, stop) - self.loss(start, split) - self.loss(split, stop)
    }

    fn find_best_split(&mut self, start: usize, stop: usize) -> usize {
        let mut max_index = 0;
        let mut max_value = -f64::INFINITY;
        let mut gain: f64;

        for index in start..stop {
            gain = self.gain(start, stop, index);
            if gain > max_value {
                max_index = index;
                max_value = gain;
            }
        }
        max_index
    }
}

pub struct ChangeInMean<'a> {
    X: &'a ndarray::Array2<f64>,
    X_cumsum: Option<ndarray::Array2<f64>>,
}

impl<'a> ChangeInMean<'a> {
    #[allow(dead_code)] // TODO Get rid of this.
    fn new(X: &'a ndarray::Array2<f64>) -> ChangeInMean<'a> {
        ChangeInMean {
            X,
            X_cumsum: Option::None,
        }
    }

    fn calculate_cumsum(&mut self) {
        let mut X_cumsum = ndarray::Array2::zeros((self.X.nrows() + 1, self.X.ncols()));
        let mut slice = X_cumsum.slice_mut(ndarray::s![1.., ..]);
        slice += &self.X.view();

        X_cumsum.accumulate_axis_inplace(ndarray::Axis(0), |&prev, curr| *curr += prev);
        self.X_cumsum = Some(X_cumsum);
    }
}

impl<'a> Segment for ChangeInMean<'a> {
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

    fn gain(&mut self, start: usize, stop: usize, split: usize) -> f64 {
        if (start == split) | (split == stop) {
            return 0.;
        }

        if self.X_cumsum.is_none() {
            self.calculate_cumsum();
        }

        let X_cumsum = self.X_cumsum.as_ref().unwrap();

        let s_1 = (split - start) as f64;
        let s_2 = (stop - split) as f64;
        let s = s_1 + s_2;

        let mut result = 0.;
        for idx in 0..self.X.ncols() {
            result += (s_1 * X_cumsum[[stop, idx]] + s_2 * X_cumsum[[start, idx]]
                - s * X_cumsum[[split, idx]])
            .powi(2)
        }
        result / (s * s_1 * s_2 * (self.X.nrows() as f64))
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use assert_approx_eq::*;
    use rstest::*;

    #[rstest]
    #[case(0, 4, 0.25)]
    #[case(0, 2, 0.)]
    #[case(0, 3, 1. / 6.)]
    #[case(1, 4, 1. / 6.)]
    #[case(1, 3, 0.125)]
    #[case(3, 3, 0.)]
    fn test_change_in_mean_loss(#[case] start: usize, #[case] stop: usize, #[case] expected: f64) {
        let X = ndarray::array![[0., 0.], [0., 0.], [0., 1.], [0., 1.]];
        assert_eq!(X.shape(), &[4, 2]);

        let change_in_mean = ChangeInMean::new(&X);
        assert_approx_eq!(change_in_mean.loss(start, stop), expected);
    }

    #[rstest]
    #[case(0, 4, 2, 0.25)]
    #[case(0, 4, 0, 0.)]
    #[case(0, 4, 1, 1. / 12.)]
    #[case(0, 4, 3, 1. / 12.)]
    #[case(0, 3, 2, 1. / 6.)]
    #[case(0, 3, 1, 1. / 24.)]
    fn test_change_in_mean_gain(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] split: usize,
        #[case] expected: f64,
    ) {
        let X = ndarray::array![[1., 0.], [1., 0.], [1., 1.], [1., 1.]];
        assert_eq!(X.shape(), &[4, 2]);

        let mut change_in_mean = ChangeInMean::new(&X);
        assert_approx_eq!(change_in_mean.gain(start, stop, split), expected);
        assert_approx_eq!(
            change_in_mean.loss(start, stop)
                - change_in_mean.loss(start, split)
                - change_in_mean.loss(split, stop),
            expected
        );
    }

    #[test]
    fn test_X_cumsum() {
        let X = ndarray::array![[1., 0.], [1., 0.], [1., 1.], [1., 1.]];
        let mut change_in_mean = ChangeInMean::new(&X);
        change_in_mean.calculate_cumsum();

        let expected = ndarray::array![[0., 0.], [1., 0.], [2., 0.], [3., 1.], [4., 2.]];
        assert_eq!(change_in_mean.X_cumsum.unwrap(), expected);
    }

    #[rstest]
    #[case(0, 7, 4)]
    #[case(1, 7, 4)]
    #[case(2, 7, 4)]
    #[case(0, 5, 4)]
    #[case(0, 2, 0)]
    fn test_change_in_mean_find_best_split(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] expected: usize,
    ) {
        let X = ndarray::array![[0.], [0.], [1.], [1.], [-1.], [-1.], [-1.]];
        assert_eq!(X.shape(), &[7, 1]);

        let mut change_in_mean = ChangeInMean::new(&X);

        assert_eq!(change_in_mean.find_best_split(start, stop), expected);
    }
}
