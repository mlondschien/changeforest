use ndarray::{Array1, Array2};
use std::clone::Clone;
use std::fmt;

#[derive(Debug, Clone)]
/// Container to hold results of the `gain_approx` method of the `GainApprox` trait.
pub struct ApproxGainResult {
    pub start: usize,
    pub stop: usize,
    pub guess: usize,
    pub gain: Array1<f64>,
    pub max_gain: Option<f64>,
    pub best_split: Option<usize>,
    pub likelihoods: Array2<f64>,
    pub predictions: Array1<f64>,
}

#[derive(Debug, Clone)]
/// Container to hold results of the `gain_full` method of the `Gain` trait.
pub struct FullGainResult {
    pub start: usize,
    pub stop: usize,
    pub max_gain: Option<f64>,
    pub best_split: Option<usize>,
    pub gain: Array1<f64>,
}

#[derive(Debug, Clone)]
pub enum GainResult {
    ApproxGainResult(ApproxGainResult),
    FullGainResult(FullGainResult),
}

impl GainResult {
    pub fn start(&self) -> usize {
        match self {
            GainResult::ApproxGainResult(result) => result.start,
            GainResult::FullGainResult(result) => result.start,
        }
    }

    pub fn stop(&self) -> usize {
        match self {
            GainResult::ApproxGainResult(result) => result.stop,
            GainResult::FullGainResult(result) => result.stop,
        }
    }

    pub fn max_gain(&self) -> Option<f64> {
        match self {
            GainResult::ApproxGainResult(result) => result.max_gain,
            GainResult::FullGainResult(result) => result.max_gain,
        }
    }

    pub fn best_split(&self) -> Option<usize> {
        match self {
            GainResult::ApproxGainResult(result) => result.best_split,
            GainResult::FullGainResult(result) => result.best_split,
        }
    }

    pub fn gain(&self) -> &Array1<f64> {
        match self {
            GainResult::ApproxGainResult(result) => &result.gain,
            GainResult::FullGainResult(result) => &result.gain,
        }
    }

    pub fn guess(&self) -> Option<usize> {
        match self {
            GainResult::ApproxGainResult(result) => Some(result.guess),
            GainResult::FullGainResult(_) => None,
        }
    }

    pub fn likelihoods(&self) -> Option<&Array2<f64>> {
        match self {
            GainResult::ApproxGainResult(result) => Some(&result.likelihoods),
            GainResult::FullGainResult(_) => None,
        }
    }

    pub fn predictions(&self) -> Option<&Array1<f64>> {
        match self {
            GainResult::ApproxGainResult(result) => Some(&result.predictions),
            GainResult::FullGainResult(_) => None,
        }
    }
}

// https://doc.rust-lang.org/rust-by-example/hello/print/print_display.html
impl fmt::Display for GainResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GainResult::ApproxGainResult(result) => write!(
                f,
                "ApproxGainResult(start={}, stop={}, guess={}, gain={})",
                result.start, result.stop, result.guess, result.gain
            ),
            GainResult::FullGainResult(result) => write!(
                f,
                "FullGainResult(start={}, stop={}, gain={})",
                result.start, result.stop, result.gain
            ),
        }
    }
}
