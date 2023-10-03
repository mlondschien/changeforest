use crate::binary_segmentation::BinarySegmentationResult;
use std::fmt::Display;

impl Display for BinarySegmentationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut max_lengths = [0; 4];
        let mut rows = _format_tree(self);
        rows.insert(
            0,
            vec![
                "".to_owned(),
                "best_split".to_owned(),
                "max_gain".to_owned(),
                "p_value".to_owned(),
            ],
        );
        for row in rows.iter() {
            for idx in 0..4 {
                if row[idx].chars().count() > max_lengths[idx] {
                    max_lengths[idx] = row[idx].chars().count();
                }
            }
        }

        for (i_row, row) in rows.iter().enumerate() {
            if i_row != 0 {
                writeln!(f)?; // write!(f, "\n")
            }
            for idx in 0..4 {
                if idx == 0 {
                    write!(f, "{}", row[idx])?;
                }

                for _ in 0..(max_lengths[idx] - row[idx].chars().count() + 1) {
                    write!(f, " ")?;
                }

                if idx != 0 {
                    write!(f, "{}", row[idx])?;
                }
            }
        }
        Ok(())
    }
}

fn _display_option<T>(option: &Option<T>) -> String
where
    T: Display,
{
    if let Some(val) = option {
        format!("{val}")
    } else {
        "".to_string()
    }
}

fn _format_tree(result: &BinarySegmentationResult) -> Vec<Vec<String>> {
    let mut output = vec![vec![
        format!("({}, {}]", result.start, result.stop),
        _display_option(&result.optimizer_result.as_ref().map(|x| x.best_split)),
        // Truncate max_gain to three decimal places.
        _display_option(
            &result
                .optimizer_result
                .as_ref()
                .map(|x| f64::trunc(x.max_gain * 1000.0) / 1000.0),
        ),
        _display_option(&result.model_selection_result.p_value),
    ]];

    if result.left.is_some() {
        let mut left = _format_tree(result.left.as_ref().unwrap());
        let mut right = _format_tree(result.right.as_ref().unwrap());
        left[0][0] = format!(" ¦--{}", left[0][0]);
        right[0][0] = format!(" °--{}", right[0][0]);

        for l in left.iter_mut().skip(1) {
            l[0] = format!(" ¦  {}", l[0]);
        }
        for r in right.iter_mut().skip(1) {
            r[0] = format!("    {}", r[0]);
        }

        output.append(&mut left);
        output.append(&mut right);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::OptimizerResult;
    use crate::ModelSelectionResult;

    #[test]
    fn test_format_binary_segmentation_result() {
        let significant_model_selection_result = ModelSelectionResult {
            p_value: Some(0.01),
            is_significant: true,
        };
        let insignificant_model_selection_result = ModelSelectionResult {
            p_value: None,
            is_significant: false,
        };

        let tree = BinarySegmentationResult {
            start: 0,
            stop: 20,
            model_selection_result: significant_model_selection_result.clone(),
            optimizer_result: Some(OptimizerResult {
                start: 0,
                stop: 20,
                best_split: 11,
                max_gain: 1234567.23456,
                gain_results: vec![],
            }),
            left: Some(Box::new(BinarySegmentationResult {
                start: 0,
                stop: 11,
                model_selection_result: significant_model_selection_result.clone(),
                optimizer_result: Some(OptimizerResult {
                    start: 0,
                    stop: 11,
                    best_split: 7,
                    max_gain: 0.1,
                    gain_results: vec![],
                }),
                left: Some(Box::new(BinarySegmentationResult {
                    start: 0,
                    stop: 7,
                    model_selection_result: significant_model_selection_result.clone(),
                    optimizer_result: None,
                    left: None,
                    right: None,
                    segments: None,
                })),
                right: Some(Box::new(BinarySegmentationResult {
                    start: 7,
                    stop: 11,
                    model_selection_result: insignificant_model_selection_result.clone(),
                    optimizer_result: None,
                    left: None,
                    right: None,
                    segments: None,
                })),
                segments: None,
            })),
            right: Some(Box::new(BinarySegmentationResult {
                start: 11,
                stop: 20,
                model_selection_result: significant_model_selection_result.clone(),
                optimizer_result: None,
                left: None,
                right: None,
                segments: None,
            })),
            segments: None,
        };

        let output = _format_tree(&tree);
        assert_eq!(
            output,
            vec![
                vec!["(0, 20]", "11", "1234567.234", "0.01"],
                vec![" ¦--(0, 11]", "7", "0.1", "0.01"],
                vec![" ¦   ¦--(0, 7]", "", "", "0.01"],
                vec![" ¦   °--(7, 11]", "", "", ""],
                vec![" °--(11, 20]", "", "", "0.01"],
            ]
        );

        let fmt = format!("\n{}", tree);
        assert_eq!(
            fmt,
            r#"
                 best_split    max_gain p_value
(0, 20]                  11 1234567.234    0.01
 ¦--(0, 11]               7         0.1    0.01
 ¦   ¦--(0, 7]                             0.01
 ¦   °--(7, 11]                                
 °--(11, 20]                               0.01"#
        );
    }
}
