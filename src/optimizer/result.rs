use ndarray::Array1;

pub struct Result {
    pub start: usize,
    pub stop: usize,
    pub gain: Array1<f64>,
    pub best_split: usize,
    pub max_gain: f64,
}
