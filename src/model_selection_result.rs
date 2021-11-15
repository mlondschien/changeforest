use std::fmt;

#[derive(Clone, Debug)]
pub struct ModelSelectionResult {
    pub is_significant: bool,
    pub p_value: Option<f64>,
}

// https://doc.rust-lang.org/rust-by-example/hello/print/print_display.html
impl fmt::Display for ModelSelectionResult {
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ModelSelectionResult(is_significant={}, p_value={:?})",
            self.is_significant, self.p_value
        )
    }
}

impl Default for ModelSelectionResult {
    fn default() -> Self {
        ModelSelectionResult {
            is_significant: false,
            p_value: None,
        }
    }
}
