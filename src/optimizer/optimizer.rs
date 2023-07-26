use crate::optimizer::OptimizerResult;
use crate::{Control, ModelSelectionResult};
use core::cmp::max;
use core::cmp::min;

pub trait Optimizer {
    /// Find the element of `split_candidates` to split segment `[start, stop)`.
    ///
    /// Returns a tuple with the best split and the maximal gain.
    fn find_best_split(&self, start: usize, stop: usize) -> Result<OptimizerResult, &str>;

    /// Does a certain split corresponds to a true change point?
    fn model_selection(&self, optimizer_result: &OptimizerResult) -> ModelSelectionResult;

    /// Total number of observations.
    fn n(&self) -> usize;

    /// Control parameters.
    fn control(&self) -> &Control;

    /// Vector with indices of allowed split points.
    fn split_candidates(&self, start: usize, stop: usize) -> Result<Vec<usize>, &str> {
        // when the user supplies nosplit_before_index or nosplit_after_index
        // we change the start and stop this way
        let actual_start = max(start, self.control().nosplit_before_index.unwrap_or(0));
        let actual_stop = min(stop, self.control().nosplit_after_index.unwrap_or(self.n()));
        let minimal_segment_length =
            (self.control().minimal_relative_segment_length * (self.n() as f64)).ceil() as usize;
        if 2 * minimal_segment_length >= (actual_stop - actual_start) {
            Err("Segment too small.")
        } else {
            Ok(((actual_start + minimal_segment_length)..(actual_stop - minimal_segment_length)).collect())
        }
    }
}
