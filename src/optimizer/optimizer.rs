use crate::optimizer::Result;

pub trait Optimizer {
    fn find_best_split(
        &self,
        start: usize,
        stop: usize,
        split_candidates: impl Iterator<Item = usize>,
    ) -> Result;
}
