use changeforest::Control;
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
            control = control.with_minimal_gain_to_split(value);
        }

        if let Some(value) = robj.dollar("model_selection_alpha").unwrap().as_real() {
            control = control.with_model_selection_alpha(value);
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
        }

        if let Some(value) = robj.dollar("random_forest_n_trees").unwrap().as_real() {
            control = control.with_random_forest_n_trees(value as usize);
        }

        Ok(MyControl { control })
    }
}
