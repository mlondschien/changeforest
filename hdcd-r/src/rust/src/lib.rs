// Allow capital X for arrays.
#![allow(non_snake_case)]

use extendr_api::prelude::*;
use hdcd::wrapper;
use hdcd::BinarySegmentationResult;
use ndarray;

// Wrapper around BinarySegmentationResult to allow implementation of extendr_api::From
// e.g. https://stackoverflow.com/questions/25413201/how-do-i-implement-a-trait-i-dont-own-for-a-type-i-dont-own
struct MyBinarySegmentationResult {
    pub result: BinarySegmentationResult,
}

// https://github.com/extendr/extendr/issues/308
impl From<MyBinarySegmentationResult> for Robj {
    fn from(my_result: MyBinarySegmentationResult) -> Self {
        let left: Robj = match my_result.result.left {
            Some(boxed_tree) => MyBinarySegmentationResult {
                result: *boxed_tree,
            }
            .into(),
            None => ().into(),
        };

        let right: Robj = match my_result.result.right {
            Some(boxed_tree) => MyBinarySegmentationResult {
                result: *boxed_tree,
            }
            .into(),
            None => ().into(),
        };

        List::from_values(&[
            r!(my_result.result.start as i32),
            r!(my_result.result.stop as i32),
            r!(my_result.result.best_split.map(|u| u as i32)),
            r!(left),
            r!(right),
        ])
        .into_robj()
        .set_names(&["start", "stop", "best_split", "left", "right"])
        .expect("From<Tree> failed")
    }
}

/// Find change points in a time series.
///
/// @param X Numerical matrix with time series.
/// @param method Either 'knn','change_in_mean' of 'random_forest'.
/// @param segmentation_type Either 'bs', 'sbs' or 'wbs'.
/// @export
#[extendr]
fn hdcd(
    X: ndarray::ArrayView2<f64>,
    method: &str,
    segmentation: &str,
) -> MyBinarySegmentationResult {
    MyBinarySegmentationResult {
        result: wrapper::hdcd(&X, method, segmentation),
    }
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod hdcdr;
    fn hdcd;
}
