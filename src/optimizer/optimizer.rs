pub trait Optimizer {
    /// Find the element of `split_candidates` to split segment `[start, stop)`.
    ///
    /// Returns a tuple with the best split and the maximal gain.
    fn find_best_split(
        &self,
        start: usize,
        stop: usize,
        split_candidates: &[usize],
    ) -> (usize, f64);
    fn is_significant(&self, start: usize, stop: usize, split: usize, max_gain: f64) -> bool;
}
