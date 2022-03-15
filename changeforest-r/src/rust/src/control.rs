use biosphere::MaxFeatures;
use changeforest::Control;
use extendr_api::Operators;
use extendr_api::Rinternals;
use extendr_api::{FromRobj, Robj};

pub struct MyControl {
    pub control: Control,
}

impl<'a> FromRobj<'a> for MyControl {
    fn from_robj(robj: &'a Robj) -> std::result::Result<Self, &'static str> {
        let mut control = Control::default();

        if let Some(value) = robj
            .dollar("minimal_relative_segment_length")
            .unwrap()
            .as_real()
        {
            control = control.with_minimal_relative_segment_length(value);
        }

        if let Some(value) = robj.dollar("minimal_gain_to_split").unwrap().as_real() {
            control = control.with_minimal_gain_to_split(Some(value));
        }

        if let Some(value) = robj.dollar("model_selection_alpha").unwrap().as_real() {
            control = control.with_model_selection_alpha(value);
        }

        if let Some(value) = robj
            .dollar("model_selection_n_permutations")
            .unwrap()
            .as_real()
        {
            control = control.with_model_selection_n_permutations(value as usize);
        }

        // as_integer does not seem to work.
        if let Some(value) = robj.dollar("number_of_wild_segments").unwrap().as_real() {
            control = control.with_number_of_wild_segments(value as usize);
        }

        if let Some(value) = robj.dollar("seeded_segments_alpha").unwrap().as_real() {
            control = control.with_seeded_segments_alpha(value);
        }

        if let Some(value) = robj.dollar("seed").unwrap().as_real() {
            control = control.with_seed(value as u64);
            control.random_forest_parameters =
                control.random_forest_parameters.with_seed(value as u64);
        }

        if let Some(value) = robj.dollar("random_forest_n_estimators").unwrap().as_real() {
            control.random_forest_parameters = control
                .random_forest_parameters
                .with_n_estimators(value as usize);
        }

        if let Some(value) = robj.dollar("random_forest_max_features").unwrap().as_real() {
            if value <= 0. {
                panic!("Got random_forest_max_features = {}", value);
            } else if value < 1. {
                control.random_forest_parameters = control
                    .random_forest_parameters
                    .with_max_features(MaxFeatures::Fraction(value));
            } else {
                control.random_forest_parameters = control
                    .random_forest_parameters
                    .with_max_features(MaxFeatures::Value(value as usize));
            }
        } else if robj.dollar("random_forest_max_features").unwrap().is_null() {
            control.random_forest_parameters = control
                .random_forest_parameters
                .with_max_features(MaxFeatures::None);
        } else if let Some(value) = robj.dollar("random_forest_max_features").unwrap().as_str() {
            if value == "sqrt" {
                control.random_forest_parameters = control
                    .random_forest_parameters
                    .with_max_features(MaxFeatures::Sqrt);
            } else if value != "default" {
                panic!("Got random_forest_max_features = {}", value);
            }
        }

        if let Some(value) = robj.dollar("random_forest_n_jobs").unwrap().as_real() {
            control.random_forest_parameters = control
                .random_forest_parameters
                .with_n_jobs(Some(value as i32));
        }

        if let Some(value) = robj.dollar("random_forest_max_depth").unwrap().as_real() {
            control.random_forest_parameters = control
                .random_forest_parameters
                .with_max_depth(Some(value as usize));
        }

        Ok(MyControl { control })
    }
}
