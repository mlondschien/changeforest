// Allow capital X for arrays.
#![allow(non_snake_case)]

use changeforest::{wrapper::changeforest, Control};
use csv::ReaderBuilder;
use ndarray::Array2;
use ndarray_csv::Array2Reader;
use rstest::*;
use std::fs::File;

#[rstest]
#[case("knn", "bs")]
#[case("knn", "wbs")]
#[case("knn", "sbs")]
#[case("change_in_mean", "bs")]
#[case("change_in_mean", "wbs")]
#[case("change_in_mean", "sbs")]
#[case("random_forest", "bs")]
#[case("random_forest", "wbs")]
#[case("random_forest", "sbs")]
fn test_integration_iris(#[case] method: &str, #[case] segmentation_type: &str) {
    let file = File::open("testdata/iris.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let X: Array2<f64> = reader.deserialize_array2((150, 4)).unwrap();

    let control = Control::default();

    let _ = changeforest(&X.view(), method, segmentation_type, &control);
}

#[rstest]
// TODO: These kill my machine.
// #[case("knn", "bs")]
// #[case("knn", "wbs")]
// #[case("knn", "sbs")]
#[case("change_in_mean", "bs")]
#[case("change_in_mean", "sbs")]
#[case("change_in_mean", "wbs")]
#[case("random_forest", "bs")]
#[case("random_forest", "sbs")]
#[case("random_forest", "wbs")]
// These are slow. Only run them with --release, i.e. cargo test --release
fn test_integration_letters(#[case] method: &str, #[case] segmentation_type: &str) {
    let file = File::open("testdata/letters.csv").unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);
    let X: Array2<f64> = reader.deserialize_array2((20000, 16)).unwrap();

    let mut control = Control::default();
    control.random_forest_parameters = control.random_forest_parameters.with_n_estimators(20);

    let _ = changeforest(&X.view(), method, segmentation_type, &control);
}
