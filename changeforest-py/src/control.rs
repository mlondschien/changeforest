use biosphere::MaxFeatures;
use changeforest::Control;
use pyo3::exceptions;
use pyo3::prelude::{pyclass, FromPyObject, PyAny, PyErr, PyResult};
use pyo3::prelude::{PyObject, Python};

pub fn control_from_pyobj(py: Python, obj: Option<PyObject>) -> PyResult<Control> {
    let mut control = Control::default();

    if let Some(obj) = obj {
        if let Ok(pyvalue) = obj.getattr(py, "minimal_relative_segment_length") {
            if let Ok(value) = pyvalue.extract::<f64>(py) {
                control = control.with_minimal_relative_segment_length(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "minimal_gain_to_split") {
            if let Ok(value) = pyvalue.extract::<Option<f64>>(py) {
                control = control.with_minimal_gain_to_split(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "model_selection_alpha") {
            if let Ok(value) = pyvalue.extract::<f64>(py) {
                control = control.with_model_selection_alpha(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "model_selection_alpha") {
            if let Ok(value) = pyvalue.extract::<f64>(py) {
                control = control.with_model_selection_alpha(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "model_selection_n_permutations") {
            if let Ok(value) = pyvalue.extract::<usize>(py) {
                control = control.with_model_selection_n_permutations(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "number_of_wild_segments") {
            if let Ok(value) = pyvalue.extract::<usize>(py) {
                control = control.with_number_of_wild_segments(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "seed") {
            if let Ok(value) = pyvalue.extract::<u64>(py) {
                control = control.with_seed(value);
                control.random_forest_parameters =
                    control.random_forest_parameters.with_seed(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "seeded_segments_alpha") {
            if let Ok(value) = pyvalue.extract::<f64>(py) {
                control = control.with_seeded_segments_alpha(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "random_forest_n_estimators") {
            if let Ok(value) = pyvalue.extract::<usize>(py) {
                control.random_forest_parameters =
                    control.random_forest_parameters.with_n_estimators(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "random_forest_max_depth") {
            if let Ok(value) = pyvalue.extract::<Option<usize>>(py) {
                control.random_forest_parameters =
                    control.random_forest_parameters.with_max_depth(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "random_forest_max_features") {
            if let Ok(value) = pyvalue.extract::<PyMaxFeatures>(py) {
                control.random_forest_parameters = control
                    .random_forest_parameters
                    .with_max_features(value.value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "random_forest_n_jobs") {
            if let Ok(value) = pyvalue.extract::<Option<i32>>(py) {
                control.random_forest_parameters =
                    control.random_forest_parameters.with_n_jobs(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "forbidden_segments") {
            if let Ok(value) = pyvalue.extract::<Option<Vec<(usize, usize)>>>(py) {
                control = control.with_forbidden_segments(value);
            }
        };
    }

    Ok(control)
}

#[pyclass(name = "MaxFeatures")]
pub struct PyMaxFeatures {
    pub value: MaxFeatures,
}

impl PyMaxFeatures {
    fn new(value: MaxFeatures) -> Self {
        PyMaxFeatures { value }
    }
}

impl FromPyObject<'_> for PyMaxFeatures {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        if let Ok(value) = ob.extract::<usize>() {
            Ok(PyMaxFeatures::new(MaxFeatures::Value(value)))
        } else if let Ok(value) = ob.extract::<f64>() {
            if value > 1. || value <= 0. {
                Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                    "Got max_features {}",
                    value
                )))
            } else {
                Ok(PyMaxFeatures::new(MaxFeatures::Fraction(value)))
            }
        } else if let Ok(value) = ob.extract::<Option<String>>() {
            if value.is_none() {
                Ok(PyMaxFeatures::new(MaxFeatures::None))
            } else {
                if value.as_ref().unwrap() == "sqrt" {
                    Ok(PyMaxFeatures::new(MaxFeatures::Sqrt))
                } else {
                    Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                        "Unknown value for max_features: {}",
                        value.unwrap()
                    )))
                }
            }
        } else {
            Err(PyErr::new::<exceptions::PyTypeError, _>(format!(
                "Unknown value for max_features: {}",
                ob
            )))
        }
    }
}
