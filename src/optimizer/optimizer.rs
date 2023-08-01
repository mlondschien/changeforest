use crate::optimizer::OptimizerResult;
use crate::{Control, ModelSelectionResult};

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
        let minimal_segment_length =
            (self.control().minimal_relative_segment_length * (self.n() as f64)).ceil() as usize;
        if 2 * minimal_segment_length >= (stop - start) {
            Err("Segment too small.")
        } else {
            // implementation suggested by @mlondschien in https://github.com/mlondschien/changeforest/pull/151 
            let mut split_candidates: Vec<usize> = ((start + minimal_segment_length)..(stop - minimal_segment_length)).collect();
            
            if let Some(forbidden_segments) = &self.control().forbidden_segments {
                split_candidates = split_candidates.into_iter().filter(|x|
                     forbidden_segments.iter().all(|segment| x < &segment.0 || x > &segment.1)
                 ).collect();
            }
            
            if split_candidates.len() == 0 {
                Err("No split_candidates left after filtering out forbidden_segments.")
            } else {
                Ok(split_candidates)
            }
        }
    }
}
