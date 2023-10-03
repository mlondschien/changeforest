pub fn log_eta(x: f64) -> f64 {
    // 1e-6 ~ 0.00247, 1 - 1e-6 ~ 0.99752
    // log_eta(1) = 0
    (0.0024787521766663585 + 0.9975212478233336 * x).ln()
}

#[cfg(test)]
mod tests {

    use super::*;
    use rstest::*;

    #[rstest]
    #[case(1., 0.)]
    #[case(0., -6.)]
    fn test_log_eta(#[case] x: f64, #[case] expected: f64) {
        assert_eq!(log_eta(x), expected);
    }
}
