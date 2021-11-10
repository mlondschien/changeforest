mod result;

use crate::result::MyBinarySegmentationResult;
use hdcd::{wrapper, Control};
use numpy::PyReadonlyArray2;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[allow(non_snake_case)] // Allow capital X for arrays.
#[pymodule]
fn hdcdpython(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn hdcd<'py>(
        X: PyReadonlyArray2<'py, f64>,
        method: String,
        segmentation_type: String,
    ) -> PyResult<MyBinarySegmentationResult> {
        let control = Control::default();
        Ok(MyBinarySegmentationResult {
            result: wrapper::hdcd(&X.as_array(), &method, &segmentation_type, &control),
        })
    }
    Ok(())
}
