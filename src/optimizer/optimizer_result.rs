use ndarray::Array1;

pub struct OptimizerResult {
    pub start: usize,
    pub stop: usize,
    pub best_split: usize,
    pub max_gain: f64,
    pub gain: Array1<f64>,
}
