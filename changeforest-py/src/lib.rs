mod control;
mod result;
use crate::control::control_from_pyobj;
use crate::result::{MyBinarySegmentationResult, MyOptimizerResult};
use ::changeforest::wrapper;
use numpy::PyReadonlyArray2;
use pyo3::prelude::{pyfunction, pymodule, wrap_pyfunction, Bound, PyModule, PyResult, Python};
use pyo3::types::PyModuleMethods;
use pyo3::Py;
use pyo3::PyAny;

#[allow(non_snake_case)] // Allow capital X for arrays.
#[pyfunction(name = "changeforest")]
#[pyo3(signature = (X, method=None, segmentation_type=None, control=None))]
fn changeforest_fn(
    py: Python<'_>,
    X: PyReadonlyArray2<f64>,
    method: Option<String>,
    segmentation_type: Option<String>,
    control: Option<Py<PyAny>>,
) -> PyResult<MyBinarySegmentationResult> {
    let control = control_from_pyobj(py, control).unwrap();
    let method = method.unwrap_or("random_forest".to_string());
    let segmentation_type = segmentation_type.unwrap_or("bs".to_string());
    Ok(MyBinarySegmentationResult {
        result: wrapper::changeforest(&X.as_array(), &method, &segmentation_type, &control),
    })
}

#[pymodule]
fn changeforest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(changeforest_fn, m)?)?;
    m.add_class::<MyBinarySegmentationResult>()?;
    m.add_class::<MyOptimizerResult>()?;
    Ok(())
}
