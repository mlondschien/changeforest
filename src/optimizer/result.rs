use ndarray::Array1;

pub struct Result {
    pub gain: Array1<f64>,
    pub best_split: usize,
    pub max_gain: f64,
    pub is_significant: bool,
}
