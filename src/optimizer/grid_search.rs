use crate::{Control, Gain, Optimizer};
use ndarray::Array1;

pub struct GridSearch<'a, T: Gain> {
    pub gain: T,
    pub control: &'a Control,
}

impl<'a, T> Optimizer for GridSearch<'a, T>
where
    T: Gain,
{
    fn n(&self) -> usize {
        self.gain.n()
    }

    fn control(&self) -> &Control {
        self.control
    }

    fn find_best_split(&self, start: usize, stop: usize) -> Result<(usize, f64), &str> {
        let split_candidates = self.split_candidates(start, stop);

        if split_candidates.is_empty() {
            return Err("Segment too small.");
        }

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

        Ok((best_split, max_gain))
    }

    fn is_significant(&self, start: usize, stop: usize, split: usize, max_gain: f64) -> bool {
        self.gain.is_significant(start, stop, split, max_gain)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::testing;
    use rstest::*;

    #[rstest]
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
        let control = Control::default().with_minimal_relative_segment_length(0.1);
        let grid_search = GridSearch {
            gain,
            control: &control,
        };
        assert_eq!(
            grid_search.find_best_split(start, stop).unwrap().0,
            expected
        );
    }
}
