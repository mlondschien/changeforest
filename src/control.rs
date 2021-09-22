#[derive(Copy, Clone)]
pub struct Control {
    pub minimal_relative_segment_length: f64,
    pub minimal_gain_to_split: f64,
}

impl Control {
    pub fn default() -> Control {
        Control {
            minimal_relative_segment_length: 0.1,
            minimal_gain_to_split: 1.,
        }
    }
}
