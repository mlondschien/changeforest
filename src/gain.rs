use super::loss;
use ndarray;

pub trait Gain {
    fn gain(&self, X: &ndarray::Array2<f64>, start: usize, stop: usize, split: usize) -> f64;
}

pub struct GainFromLoss {
    pub loss: Box<dyn loss::Loss>,
}

impl Gain for GainFromLoss {
    fn gain(&self, X: &ndarray::Array2<f64>, start: usize, stop: usize, split: usize) -> f64 {
        self.loss.loss(X, start, stop) // X already is a reference here (?)
            - self.loss.loss(X, start, split)
            - self.loss.loss(X, split, stop)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use assert_approx_eq::*;
    use rstest::*;

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
        let X = ndarray::array![[0., 0.], [0., 0.], [0., 1.], [0., 1.]];
        assert_eq!(X.shape(), &[4, 2]);

        let gain = GainFromLoss {
            loss: Box::new(loss::ChangeInMeanLoss {}),
        };
        assert_approx_eq!(gain.gain(&X, start, stop, split), expected);
    }
}
