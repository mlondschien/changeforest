use super::gain;
use ndarray;

pub trait Optimizer {
    fn find_best_split(&self, X: &ndarray::Array2<f64>, start: usize, stop: usize) -> usize;
}

pub struct GridSearchOptimizer {
    gain: Box<dyn gain::Gain>,
}

impl Optimizer for GridSearchOptimizer {
    fn find_best_split(&self, X: ndarray::Array2<f64>, start: usize, stop: usize) -> usize {
        let mut max_index = 0;
        let mut max_value = -f64::INFINITY;
        let mut gain: f64;

        for index in start..stop {
            gain = self.gain.gain(start, stop, index);
            if gain > max_value {
                max_index = index;
                max_value = gain;
            }
        }
        max_index
    }
}

#[cfg(test)]
mod tests {

    use super::super::gain;
    use super::*;
    use rstest::*;

    #[rstest]
    #[case(0, 7, 4)]
    #[case(1, 7, 4)]
    #[case(2, 7, 4)]
    #[case(0, 5, 4)]
    #[case(0, 2, 0)]
    fn test_best_split_change_in_mean(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] expected: usize,
    ) {
        let X = ndarray::array![[0.], [0.], [1.], [1.], [-1.], [-1.], [-1.]];
        assert_eq!(X.shape(), &[7, 1]);

        let change_in_mean = Box::new(gain::ChangeInMean { X: &X });

        let grid_search_optimizer = GridSearchOptimizer {
            gain: change_in_mean,
        };
        assert_eq!(
            grid_search_optimizer.find_best_split(&X, start, stop),
            expected
        );
    }
}
