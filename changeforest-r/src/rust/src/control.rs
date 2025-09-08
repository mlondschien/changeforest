use biosphere::MaxFeatures;
use changeforest::Control;
use extendr_api::prelude::*;

#[derive(Debug, Clone)]
pub struct MyControl {
    pub control: Control,
}

impl TryFromRobj for MyControl {
    type Error = extendr_api::Error;

    fn try_from_robj(robj: &Robj) -> Result<Self, Self::Error> {
        let mut control = Control::default();

        // Helper function to safely get optional values
        let get_real_option =
            |obj: &Robj, name: &str| -> Option<f64> { obj.dollar(name).ok()?.as_real() };

        let get_str_option = |obj: &Robj, name: &str| -> Option<String> {
            obj.dollar(name).ok()?.as_str().map(|s| s.to_string())
        };

        if let Some(value) = get_real_option(robj, "minimal_relative_segment_length") {
            control = control.with_minimal_relative_segment_length(value);
        }

        if let Some(value) = get_real_option(robj, "minimal_gain_to_split") {
            control = control.with_minimal_gain_to_split(Some(value));
        }

        if let Some(value) = get_real_option(robj, "model_selection_alpha") {
            control = control.with_model_selection_alpha(value);
        }

        if let Some(value) = get_real_option(robj, "model_selection_n_permutations") {
            control = control.with_model_selection_n_permutations(value as usize);
        }

        if let Some(value) = get_real_option(robj, "number_of_wild_segments") {
            control = control.with_number_of_wild_segments(value as usize);
        }

        if let Some(value) = get_real_option(robj, "seeded_segments_alpha") {
            control = control.with_seeded_segments_alpha(value);
        }

        if let Some(value) = get_real_option(robj, "seed") {
            let seed = value as u64;
            control = control.with_seed(seed);
            control.random_forest_parameters = control.random_forest_parameters.with_seed(seed);
        }

        if let Some(value) = get_real_option(robj, "random_forest_n_estimators") {
            control.random_forest_parameters = control
                .random_forest_parameters
                .with_n_estimators(value as usize);
        }

        // Handle max_features (can be real, string, or null)
        if let Ok(max_features_obj) = robj.dollar("random_forest_max_features") {
            if max_features_obj.is_null() {
                control.random_forest_parameters = control
                    .random_forest_parameters
                    .with_max_features(MaxFeatures::None);
            } else if let Some(value) = max_features_obj.as_real() {
                if value <= 0. {
                    return Err(extendr_api::Error::Other(format!(
                        "Got random_forest_max_features = {value}"
                    )));
                } else if value < 1. {
                    control.random_forest_parameters = control
                        .random_forest_parameters
                        .with_max_features(MaxFeatures::Fraction(value));
                } else {
                    control.random_forest_parameters = control
                        .random_forest_parameters
                        .with_max_features(MaxFeatures::Value(value as usize));
                }
            } else if let Some(value) = max_features_obj.as_str() {
                match value {
                    "sqrt" => {
                        control.random_forest_parameters = control
                            .random_forest_parameters
                            .with_max_features(MaxFeatures::Sqrt);
                    }
                    "default" => {
                        // Keep default, do nothing
                    }
                    _ => {
                        return Err(extendr_api::Error::Other(format!(
                            "Got random_forest_max_features = {value}"
                        )));
                    }
                }
            }
        }

        if let Some(value) = get_real_option(robj, "random_forest_n_jobs") {
            control.random_forest_parameters = control
                .random_forest_parameters
                .with_n_jobs(Some(value as i32));
        }

        if let Some(value) = get_real_option(robj, "random_forest_max_depth") {
            control.random_forest_parameters = control
                .random_forest_parameters
                .with_max_depth(Some(value as usize));
        }

        Ok(MyControl { control })
    }
}
