// Allow capital X for arrays.
#![allow(non_snake_case)]

mod control;
mod result;

use crate::control::MyControl;
use crate::result::MyBinarySegmentationResult;
use changeforest::wrapper;
use extendr_api::prelude::*;
use ndarray;

#[extendr]
fn changeforest_api(
    X: ndarray::ArrayView2<f64>,
    method: &str,
    segmentation: &str,
    control: MyControl,
) -> MyBinarySegmentationResult {
    MyBinarySegmentationResult {
        result: wrapper::changeforest(&X, method, segmentation, &control.control),
    }
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod changeforestr;
    fn changeforest_api;
}
