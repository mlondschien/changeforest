// Allow capital X for arrays.
#![allow(non_snake_case)]

use extendr_api::prelude::*;
use hdcd::wrapper;
use ndarray;

/// Find change points in a time series.
/// @export
#[extendr]
fn hdcd(X: ndarray::ArrayView2<f64>) -> Vec<usize> {
    wrapper::hdcd(&X)
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    fn hdcd;
    mod hdcd;
}
