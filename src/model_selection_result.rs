use std::fmt;

#[derive(Clone, Debug, Default)]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_selection_result_default() {
        // Clippy complains if I implement default myself. It is very essential for
        // this crate / the algorithm that `is_significant` is initialized as `false`.
        let result = ModelSelectionResult::default();
        assert!(!result.is_significant);
    }
}
