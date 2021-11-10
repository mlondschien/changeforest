// Wrap GainResult, OptimizerResult and BinarySegmentationResult.
// See https://github.com/PyO3/pyo3/issues/287.

use hdcd::gain::GainResult;
use hdcd::optimizer::OptimizerResult;
use hdcd::BinarySegmentationResult;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3;
use pyo3::prelude::*; //{pymodule, getter, pymethods, pyclass, PyModule, PyResult, Python};
use pyo3::{class, Python};

#[pyclass]
#[derive(Clone, Debug)]
pub struct MyGainResult {
    pub result: GainResult,
}

#[pymethods]
impl MyGainResult {
    #[getter]
    pub fn start(&self) -> usize {
        self.result.start()
    }

    #[getter]
    pub fn stop(&self) -> usize {
        self.result.stop()
    }

    #[getter]
    pub fn gain<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
        self.result.gain().to_pyarray(py)
    }

    #[getter]
    pub fn guess(&self) -> Option<usize> {
        self.result.guess()
    }

    #[getter]
    pub fn likelihoods<'py>(&self, py: Python<'py>) -> Option<&'py PyArray2<f64>> {
        self.result.likelihoods().map(|arr| arr.to_pyarray(py))
    }

    #[getter]
    pub fn predictions<'py>(&self, py: Python<'py>) -> Option<&'py PyArray1<f64>> {
        self.result.predictions().map(|arr| arr.to_pyarray(py))
    }
}

#[pyproto]
impl pyo3::class::basic::PyObjectProtocol for MyGainResult {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.result))
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MyOptimizerResult {
    pub result: OptimizerResult,
}

#[pymethods]
impl MyOptimizerResult {
    #[getter]
    fn start(&self) -> usize {
        self.result.start
    }

    #[getter]
    fn stop(&self) -> usize {
        self.result.stop
    }

    #[getter]
    fn best_split(&self) -> usize {
        self.result.best_split
    }

    #[getter]
    fn max_gain(&self) -> f64 {
        self.result.max_gain
    }

    #[getter]
    fn gain_results(&self) -> Vec<MyGainResult> {
        self.result
            .gain_results
            .iter()
            .map(|result| MyGainResult {
                result: result.clone(),
            })
            .collect()
    }
}

#[pyproto]
// https://stackoverflow.com/questions/62666926/str-function-of-class-ported-from-\
// rust-to-python-using-pyo3-doesnt-get-used
// https://pyo3.rs/v0.9.2/python_from_rust.html
impl pyo3::class::basic::PyObjectProtocol for MyOptimizerResult {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.result))
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct MyBinarySegmentationResult {
    pub result: BinarySegmentationResult,
}

#[pymethods]
impl MyBinarySegmentationResult {
    #[getter]
    fn start(&self) -> usize {
        self.result.start
    }

    #[getter]
    fn stop(&self) -> usize {
        self.result.stop
    }

    #[getter]
    fn best_split(&self) -> Option<usize> {
        self.result.best_split
    }

    #[getter]
    fn max_gain(&self) -> Option<f64> {
        self.result.max_gain
    }

    #[getter]
    fn gain_results(&self) -> Option<Vec<MyGainResult>> {
        self.result.gain_results.as_ref().map(|results| {
            results
                .iter()
                .map(|result| MyGainResult {
                    result: result.clone(),
                })
                .collect()
        })
    }

    #[getter]
    fn left(&self) -> Option<Self> {
        // This seems overly complicated. We turn Option<Box<Self>> into
        // Option<&Box<Self>>, then clone the &Box<Self> (including the box) and deref
        // it to get Self.
        self.result
            .left
            .as_ref()
            .map(|left| MyBinarySegmentationResult {
                result: *(left.clone()),
            })
    }

    #[getter]
    fn right(&self) -> Option<Self> {
        self.result
            .right
            .as_ref()
            .map(|right| MyBinarySegmentationResult {
                result: *(right.clone()),
            })
    }

    fn split_points(&self) -> Vec<usize> {
        self.result.split_points()
    }
}
