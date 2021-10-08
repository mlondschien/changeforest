// Allow capital X for arrays.
#![allow(non_snake_case)]

mod classifier;
mod control;
mod gain;
mod optimizer;
pub use classifier::Classifier;
pub use control::Control;
pub use gain::Gain;
pub use optimizer::Optimizer;

pub mod binary_segmentation;
pub mod model_selection;
pub mod utils;
pub mod wrapper;

#[cfg(test)]
mod testing;
