use ndarray;

pub trait Loss {
    fn loss(&self, X: &ndarray::Array2<f64>, start: usize, stop: usize) -> f64;
}

pub struct ChangeInMeanLoss {}

impl Loss for ChangeInMeanLoss {
    fn loss(&self, X: &ndarray::Array2<f64>, start: usize, stop: usize) -> f64 {
        if start == stop {return 0.};

        let n_total = X.nrows() as f64;
        let n_slice = (stop - start) as f64;

        let slice = X.slice(ndarray::s![start..stop, ..]);

        // For 1D, the change in mean loss is equal to
        // 1 / n_total * [sum_i x_i**2 - 1/n_slice (sum_i x_i)**2]
        // For 2D, the change in mean loss is just the sum of losses for each dimension.
        let loss = slice.mapv(|a| a.powi(2)).sum()
            - slice.sum_axis(ndarray::Axis(0)).mapv(|a| a.powi(2)).sum() / n_slice;

        loss / n_total
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

        let change_in_mean_loss = ChangeInMeanLoss {};
        assert_approx_eq!(change_in_mean_loss.loss(&X, start, stop), expected);
    }
}
