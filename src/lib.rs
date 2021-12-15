// Allow capital X for arrays.
#![allow(non_snake_case)]
#![allow(clippy::module_inception)]
// BS, SBS and WBS
#![allow(clippy::upper_case_acronyms)]

mod binary_segmentation;
pub mod classifier;
mod control;
mod fmt;
pub mod gain;
mod model_selection_result;
pub mod optimizer;
mod segmentation;

pub use binary_segmentation::{BinarySegmentationResult, BinarySegmentationTree};
pub use classifier::Classifier;
pub use control::Control;
pub use gain::{ClassifierGain, Gain};
pub use model_selection_result::ModelSelectionResult;
pub use optimizer::Optimizer;
pub use segmentation::{Segmentation, SegmentationType};
pub mod utils;
pub mod wrapper;

#[cfg(test)]
mod testing;
