// Allow capital X for arrays.
#![allow(non_snake_case)]
#![allow(clippy::module_inception)]
// BS, SBS and WBS
#![allow(clippy::upper_case_acronyms)]

mod classifier;
mod control;
mod gain;
mod optimizer;
mod segmentation;

pub use classifier::Classifier;
pub use control::Control;
pub use gain::Gain;
pub use optimizer::Optimizer;

pub mod binary_segmentation;
pub mod utils;
pub mod wrapper;

#[cfg(test)]
mod testing;
