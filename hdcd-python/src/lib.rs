use numpy::PyReadonlyArray2;
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
//use ndarray::ArrayView2;
use hdcd::wrapper;

#[allow(non_snake_case)] // Allow capital X for arrays.
#[pymodule]
fn hdcdpython(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn hdcd<'py>(X: PyReadonlyArray2<'py, f64>) -> PyResult<Vec<usize>> {
        Ok(wrapper::hdcd(&X.as_array()))
    }
    Ok(())
}
