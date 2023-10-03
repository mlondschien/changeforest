// Wrap GainResult, OptimizerResult and BinarySegmentationResult.
// See https://github.com/PyO3/pyo3/issues/287.

use ::changeforest::{BinarySegmentationResult, ModelSelectionResult};
use changeforest::gain::GainResult;
use changeforest::optimizer::OptimizerResult;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;

#[pyclass(name = "ModelSelectionResult")]
#[derive(Clone, Debug)]
pub struct MyModelSelectionResult {
    pub result: ModelSelectionResult,
}

#[pymethods]
impl MyModelSelectionResult {
    #[getter]
    pub fn is_significant(&self) -> bool {
        self.result.is_significant
    }

    #[getter]
    pub fn p_value(&self) -> Option<f64> {
        self.result.p_value
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.result))
    }
}

#[pyclass(name = "GainResult")]
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

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.result))
    }
}

#[pyclass(name = "OptimizerResult")]
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

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.result))
    }
}

#[pyclass(name = "BinarySegmentationResult")]
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
        self.result
            .optimizer_result
            .as_ref()
            .map(|result| result.best_split)
    }

    #[getter]
    fn max_gain(&self) -> Option<f64> {
        self.result
            .optimizer_result
            .as_ref()
            .map(|result| result.max_gain)
    }

    #[getter]
    fn p_value(&self) -> Option<f64> {
        self.result.model_selection_result.p_value
    }

    #[getter]
    fn is_significant(&self) -> bool {
        self.result.model_selection_result.is_significant
    }

    #[getter]
    fn optimizer_result(&self) -> Option<MyOptimizerResult> {
        self.result
            .optimizer_result
            .as_ref()
            .map(|result| MyOptimizerResult {
                result: result.clone(),
            })
    }

    #[getter]
    fn model_selection_result(&self) -> MyModelSelectionResult {
        MyModelSelectionResult {
            result: self.result.model_selection_result.clone(),
        }
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

    #[getter]
    fn segments(&self) -> Option<Vec<MyOptimizerResult>> {
        self.result.segments.as_ref().map(|segments| {
            segments
                .iter()
                .map(|result| MyOptimizerResult {
                    result: result.clone(),
                })
                .collect()
        })
    }

    fn split_points(&self) -> Vec<usize> {
        self.result.split_points()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.result))
    }
}

#[pymodule]
fn my_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MyBinarySegmentationResult>()?;
    Ok(())
}
