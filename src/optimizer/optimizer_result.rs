use crate::gain::GainResult;
use std::fmt;

#[derive(Clone, Debug)]
pub struct OptimizerResult {
    pub start: usize,
    pub stop: usize,
    pub best_split: usize,
    pub max_gain: f64,
    pub gain_results: Vec<GainResult>,
}

// https://doc.rust-lang.org/rust-by-example/hello/print/print_display.html
impl fmt::Display for OptimizerResult {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "OptimizerResult(start={}, stop={}, best_split={}, max_gain={})",
            self.start, self.stop, self.best_split, self.max_gain
        )
    }
}
