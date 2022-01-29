use changeforest::Control;
use pyo3::prelude::{PyObject, PyResult, Python};

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
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "random_forest_n_trees") {
            if let Ok(value) = pyvalue.extract::<usize>(py) {
                control = control.with_random_forest_n_trees(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "random_forest_max_depth") {
            if let Ok(value) = pyvalue.extract::<Option<usize>>(py) {
                control = control.with_random_forest_max_depth(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "random_forest_mtry") {
            if let Ok(value) = pyvalue.extract::<Option<usize>>(py) {
                control = control.with_random_forest_mtry(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "random_forest_n_jobs") {
            if let Ok(value) = pyvalue.extract::<Option<usize>>(py) {
                control = control.with_random_forest_n_jobs(value);
            }
        };

        if let Ok(pyvalue) = obj.getattr(py, "seeded_segments_alpha") {
            if let Ok(value) = pyvalue.extract::<f64>(py) {
                control = control.with_seeded_segments_alpha(value);
            }
        };
    }

    Ok(control)
}
