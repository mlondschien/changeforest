use crate::gain::{ApproxGain, GainResult};
use crate::optimizer::OptimizerResult;
use crate::{Control, Gain, ModelSelectionResult, Optimizer};

pub struct TwoStepSearch<T: Gain> {
    pub gain: T,
}

impl<T> TwoStepSearch<T>
where
    T: Gain + ApproxGain,
{
    fn _single_find_best_split(
        &self,
        start: usize,
        stop: usize,
        guess: usize,
        split_candidates: &[usize],
    ) -> (GainResult, usize, f64) {
        let approx_gain_result = self.gain.gain_approx(start, stop, guess, split_candidates);

        let mut best_split = guess;
        let mut max_gain = -f64::INFINITY;

        for index in split_candidates {
            if approx_gain_result.gain[*index - start] > max_gain {
                best_split = *index;
                max_gain = approx_gain_result.gain[*index - start];
            }
        }

        (
            GainResult::ApproxGainResult(approx_gain_result),
            best_split,
            max_gain,
        )
    }
}

impl<T> Optimizer for TwoStepSearch<T>
where
    T: ApproxGain + Gain,
{
    fn n(&self) -> usize {
        self.gain.n()
    }

    fn control(&self) -> &Control {
        self.gain.control()
    }

    fn find_best_split(&self, start: usize, stop: usize) -> Result<OptimizerResult, &str> {
        let split_candidates = self.split_candidates(start, stop);

        if split_candidates.is_empty() {
            return Err("Segment too small.");
        }

        let (left_gain_result, left_best_split, left_max_gain) =
            self._single_find_best_split(start, stop, (3 * start + stop) / 4, &split_candidates);
        let (mid_gain_result, mid_best_split, mid_max_gain) =
            self._single_find_best_split(start, stop, (start + stop) / 2, &split_candidates);
        let (right_gain_result, right_best_split, right_max_gain) =
            self._single_find_best_split(start, stop, (start + 3 * stop) / 4, &split_candidates);

        let best_split: usize;
        if mid_max_gain >= left_max_gain && mid_max_gain >= right_max_gain {
            best_split = mid_best_split;
        } else if left_max_gain >= mid_max_gain && left_max_gain >= right_max_gain {
            best_split = left_best_split;
        } else {
            best_split = right_best_split;
        }

        let (second_gain_result, second_best_split, second_max_gain) =
            self._single_find_best_split(start, stop, best_split, &split_candidates);

        Ok(OptimizerResult {
            start,
            stop,
            best_split: second_best_split,
            max_gain: second_max_gain,
            gain_results: vec![
                left_gain_result,
                mid_gain_result,
                right_gain_result,
                second_gain_result,
            ],
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
    #[case(0, 7, 4)]
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
        let control = Control::default().with_minimal_relative_segment_length(0.01);

        let gain = testing::ChangeInMean::new(&X_view, &control);
        let two_step_search = TwoStepSearch { gain };

        assert_eq!(
            two_step_search
                .find_best_split(start, stop)
                .unwrap()
                .best_split,
            expected
        );
    }
}
