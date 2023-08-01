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
    ) -> GainResult {
        let mut approx_gain_result = self.gain.gain_approx(start, stop, guess, split_candidates);

        let mut best_split = guess;
        let mut max_gain = -f64::INFINITY;

        for index in split_candidates {
            if approx_gain_result.gain[*index - start] > max_gain {
                best_split = *index;
                max_gain = approx_gain_result.gain[*index - start];
            }
        }

        approx_gain_result.best_split = Some(best_split);
        approx_gain_result.max_gain = Some(max_gain);

        GainResult::ApproxGainResult(approx_gain_result)
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
        let split_candidates = self.split_candidates(start, stop)?;

        let mut guesses = vec![];
        let mut results: Vec<GainResult> = vec![];

        // if there are forbidden segments change the heuristics
        // pick middle element of split_candidates, 1/4th and 3/4th
        if let Some(_forbidden_segments) = &self.control().forbidden_segments {
            // there is at least one element in split_candidates
            guesses.push(
                split_candidates
                    .clone()
                    .into_iter()
                    .nth(split_candidates.len() / 4)
                    .unwrap(),
            );

            // we add this if it is not equal to last
            let cand = split_candidates
                .clone()
                .into_iter()
                .nth(split_candidates.len() / 2)
                .unwrap();
            if cand > guesses[guesses.len() - 1] {
                guesses.push(cand)
            };

            // same
            let cand = split_candidates
                .clone()
                .into_iter()
                .nth(3 * split_candidates.len() / 4)
                .unwrap();
            if cand > guesses[guesses.len() - 1] {
                guesses.push(cand)
            };
        } else {
            guesses.push((3 * start + stop) / 4);
            guesses.push((start + stop) / 2);
            guesses.push((start + 3 * stop) / 4);
        }

        // Don't use first and last guess if stop - start / 4 < delta.
        for guess in guesses.iter().filter(|x| split_candidates.contains(x)) {
            results.push(self._single_find_best_split(start, stop, *guess, &split_candidates));
        }

        let max_gain = results
            .iter()
            .map(|x| x.max_gain().unwrap())
            .reduce(f64::max)
            .unwrap();
        let best_split = results
            .iter()
            .find(|x| x.max_gain().unwrap() >= max_gain)
            .unwrap()
            .best_split()
            .unwrap();

        results.push(self._single_find_best_split(start, stop, best_split, &split_candidates));

        Ok(OptimizerResult {
            start,
            stop,
            best_split: results.last().unwrap().best_split().unwrap(),
            max_gain: results.last().unwrap().max_gain().unwrap(),
            gain_results: results,
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

    #[rstest]
    #[case(0, 100, 0.1, vec![25, 50, 75])]
    #[case(0, 100, 0.01, vec![25, 50, 75])]
    #[case(0, 30, 0.1, vec![15])]
    #[case(0, 30, 0.01, vec![7, 15, 22])]
    #[case(0, 100, 0.49, vec![50])]
    fn test_change_in_mean_find_best_split_guesses(
        #[case] start: usize,
        #[case] stop: usize,
        #[case] minimal_relative_segment_lengh: f64,
        #[case] expected: Vec<usize>,
    ) {
        let X = testing::array();
        let X_view = X.view();
        let control =
            Control::default().with_minimal_relative_segment_length(minimal_relative_segment_lengh);
        let gain = testing::ChangeInMean::new(&X_view, &control);
        let two_step_search = TwoStepSearch { gain };

        let results = two_step_search
            .find_best_split(start, stop)
            .unwrap()
            .gain_results;
        let guesses = results
            .iter()
            .map(|x| x.guess().unwrap())
            .collect::<Vec<usize>>();
        let mut guesses = guesses.split_last().unwrap().1.to_owned();
        guesses.sort();
        assert_eq!(guesses, expected);
    }
}
