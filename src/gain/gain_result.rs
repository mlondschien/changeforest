use ndarray::{Array1, Array2};

#[derive(Debug)]
/// Container to hold results of the `gain_approx` method of the `GainApprox` trait.
pub struct ApproxGainResult {
    pub start: usize,
    pub stop: usize,
    pub guess: usize,
    pub gain: Array1<f64>,
    pub likelihoods: Array2<f64>,
    pub predictions: Array1<f64>,
}

/// Container to hold results of the `gain_full` method of the `Gain` trait.
pub struct FullGainResult {
    pub start: usize,
    pub stop: usize,
    pub gain: Array1<f64>,
}

pub enum GainResult {
    ApproxGainResult(ApproxGainResult),
    FullGainResult(FullGainResult),
}
