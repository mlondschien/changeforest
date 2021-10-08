use crate::optimizer::Result;
use crate::{Gain, Optimizer};

pub struct TwoStepSearch<T: Gain> {
    pub gain: T,
}

impl<T> Optimizer for TwoStepSearch<T>
where
    T: Gain,
{
    fn find_best_split(&self, start: usize, stop: usize, split_candidates: &[usize]) -> Result {
        let gain = self
            .gain
            .gain_approx(start, stop, (start + stop) / 2, split_candidates);
        let mut best_split = 0;
        let mut max_gain = -f64::INFINITY;

        for index in split_candidates {
            if gain[*index - start] > max_gain {
                best_split = *index;
                max_gain = gain[*index - start];
            }
        }

        let gain = self
            .gain
            .gain_approx(start, stop, best_split + start, split_candidates);
        max_gain = -f64::INFINITY;
        for index in split_candidates {
            if gain[*index - start] > max_gain {
                best_split = *index;
                max_gain = gain[*index - start];
            }
        }

        Result {
            start,
            stop,
            gain,
            best_split,
            max_gain,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::testing;
    use rstest::*;

    #[rstest]
    #[case(0, 0, 0)]
    #[case(0, 7, 2)]
    #[case(1, 7, 4)]
    #[case(2, 7, 4)]
    #[case(3, 7, 4)]
    #[case(1, 5, 2)]
    #[case(1, 6, 4)]
    #[case(1, 7, 4)]
    #[case(2, 6, 4)]
    #[case(2, 7, 4)]
    #[case(3, 6, 4)]
    fn test_change_in_mean_find_best_split(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] expected: usize,
    ) {
        let X = ndarray::array![
            [0., 1.],
            [0., 1.],
            [1., -1.],
            [1., -1.],
            [-1., -1.],
            [-1., -1.],
            [-1., -1.]
        ];
        let X_view = X.view();
        assert_eq!(X_view.shape(), &[7, 2]);

        let gain = testing::ChangeInMean::new(&X_view);
        let grid_search = TwoStepSearch { gain };

        let split_points: Vec<usize> = (start..stop).collect();
        assert_eq!(
            grid_search
                .find_best_split(start, stop, &split_points)
                .best_split,
            expected
        );
    }
}
