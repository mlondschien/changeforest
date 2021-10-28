use crate::optimizer::OptimizerResult;
use crate::Control;

pub trait Optimizer {
    /// Find the element of `split_candidates` to split segment `[start, stop)`.
    ///
    /// Returns a tuple with the best split and the maximal gain.
    fn find_best_split(&self, start: usize, stop: usize) -> Result<OptimizerResult, &str>;

    /// Does a certain split corresponds to a true change point?
    fn is_significant(&self, start: usize, stop: usize, split: usize, max_gain: f64) -> bool;

    /// Total number of observations.
    fn n(&self) -> usize;

    /// Control parameters.
    fn control(&self) -> &Control;

    /// Vector with indices of allowed split points.
    fn split_candidates(&self, start: usize, stop: usize) -> Vec<usize> {
        let minimal_segment_length =
            (self.control().minimal_relative_segment_length * (self.n() as f64)).round() as usize;
        if 2 * minimal_segment_length > (stop - start) {
            vec![]
        } else {
            ((start + minimal_segment_length)..(stop - minimal_segment_length)).collect()
        }
    }
}
