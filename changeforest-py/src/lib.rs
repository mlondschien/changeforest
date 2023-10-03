mod control;
mod result;

use crate::control::control_from_pyobj;
use crate::result::{MyBinarySegmentationResult, MyOptimizerResult};
use ::changeforest::wrapper;
use numpy::PyReadonlyArray2;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use pyo3::PyObject;

// Note: This has to match the lib.name in Cargo.toml.
#[allow(non_snake_case)] // Allow capital X for arrays.
#[pymodule]
fn changeforest(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn changeforest(
        py: Python<'_>,
        X: PyReadonlyArray2<f64>,
        method: Option<String>,
        segmentation_type: Option<String>,
        control: Option<PyObject>,
    ) -> PyResult<MyBinarySegmentationResult> {
        let control = control_from_pyobj(py, control).unwrap();
        let method = method.unwrap_or("random_forest".to_string());
        let segmentation_type = segmentation_type.unwrap_or("bs".to_string());
        Ok(MyBinarySegmentationResult {
            result: wrapper::changeforest(&X.as_array(), &method, &segmentation_type, &control),
        })
    }

    m.add_class::<MyBinarySegmentationResult>()?;
    m.add_class::<MyOptimizerResult>()?;
    Ok(())
}
