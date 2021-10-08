use crate::optimizer::Result;
use crate::{Gain, Optimizer};
use ndarray::Array1;

pub struct GridSearch<T: Gain> {
    pub gain: T,
}

impl<T> Optimizer for GridSearch<T>
where
    T: Gain,
{
    fn find_best_split(
        &self,
        start: usize,
        stop: usize,
        split_candidates: impl Iterator<Item = usize>,
    ) -> Result {
        let mut gain = Array1::from_elem(stop - start, f64::NAN);

        let mut best_split = 0;
        let mut max_gain = -f64::INFINITY;

        for index in split_candidates {
            gain[index - start] = self.gain.gain(start, stop, index);
            if gain[index - start] > max_gain {
                best_split = index;
                max_gain = gain[index - start];
            }
        }

        Result {
            gain,
            best_split,
            max_gain,
            is_significant: max_gain > 0.1,
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
        let grid_search = GridSearch { gain };

        assert_eq!(
            grid_search
                .find_best_split(start, stop, start..stop)
                .best_split,
            expected
        );
    }
}
