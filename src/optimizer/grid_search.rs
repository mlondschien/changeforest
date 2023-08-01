use crate::gain::GainResult;
use crate::optimizer::OptimizerResult;
use crate::{Control, Gain, ModelSelectionResult, Optimizer};

pub struct GridSearch<T: Gain> {
    pub gain: T,
}

impl<T> Optimizer for GridSearch<T>
where
    T: Gain,
{
    fn n(&self) -> usize {
        self.gain.n()
    }

    fn control(&self) -> &Control {
        self.gain.control()
    }

    fn find_best_split(&self, start: usize, stop: usize) -> Result<OptimizerResult, &str> {
        let split_candidates = self.split_candidates(start, stop)?;

        let mut full_gain = self.gain.gain_full(start, stop, &split_candidates);

        let mut best_split = 0;
        let mut max_gain = -f64::INFINITY;

        for index in split_candidates {
            if full_gain.gain[index - start] > max_gain {
                best_split = index;
                max_gain = full_gain.gain[index - start];
            }
        }

        full_gain.max_gain = Some(max_gain);
        full_gain.best_split = Some(best_split);

        Ok(OptimizerResult {
            start,
            stop,
            best_split,
            max_gain,
            gain_results: vec![GainResult::FullGainResult(full_gain)],
        })
    }

    fn model_selection(&self, optimizer_result: &OptimizerResult) -> ModelSelectionResult {
        self.gain.model_selection(optimizer_result)
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
        let control = Control::default().with_minimal_relative_segment_length(0.1);

        let gain = testing::ChangeInMean::new(&X_view, &control);
        let grid_search = GridSearch { gain };
        assert_eq!(
            grid_search.find_best_split(start, stop).unwrap().best_split,
            expected
        );
    }

    #[rstest]
    #[case(0, 10, Some(vec![(0, 3)]), 0.09, vec![4, 5, 6, 7, 8])]
    #[case(1, 10, Some(vec![(6, 10)]), 0.15, vec![3, 4, 5, 6])]
    #[case(0, 10, Some(vec![(2, 4), (5, 7)]), 0.09, vec![1, 2, 5, 8])]
    #[case(1, 7, Some(vec![(2, 4), (5, 7)]), 0.09, vec![2, 5])]
    fn test_split_candidates(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] forbidden_segments: Option<Vec<(usize, usize)>>,
        #[case] delta: f64,
        #[case] expected: Vec<usize>,
    ) {
        let X = ndarray::array![
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [-0.0],
            [-0.0],
            [-0.0],
            [-0.0],
            [-0.0],
            [-0.0]
        ];
        let X_view = X.view();
        let control = Control::default()
            .with_minimal_relative_segment_length(delta)
            .with_forbidden_segments(forbidden_segments);
        let gain = testing::ChangeInMean::new(&X_view, &control);
        let grid_search = GridSearch { gain };
        assert_eq!(grid_search.split_candidates(start, stop).unwrap(), expected);
    }
}
