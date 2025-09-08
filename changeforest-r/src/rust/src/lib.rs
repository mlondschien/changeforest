#![allow(non_snake_case)]
mod control;
mod result;
use crate::control::MyControl;
use crate::result::MyBinarySegmentationResult;
use ::ndarray;
use changeforest::wrapper;
use extendr_api::prelude::*;
use std::convert::TryFrom;
use std::panic;

#[extendr]
fn changeforest_api(
    X: ndarray::ArrayView2<f64>,
    method: &str,
    segmentation: &str,
    control: Robj,
) -> extendr_api::Result<MyBinarySegmentationResult> {
    panic::set_hook(Box::new(|_| {
        // Do nothing on panic instead of calling exit
    }));
    // Convert control using the standard TryFrom trait
    let control = MyControl::try_from(&control)?;

    Ok(MyBinarySegmentationResult {
        result: wrapper::changeforest(&X, method, segmentation, &control.control),
    })
}

// Macro to generate exports.
extendr_module! {
    mod changeforest;
    fn changeforest_api;
}
