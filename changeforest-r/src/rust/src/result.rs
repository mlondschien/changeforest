use changeforest::gain::GainResult;
use changeforest::optimizer::OptimizerResult;
use changeforest::{BinarySegmentationResult, ModelSelectionResult};
use extendr_api::prelude::*;

pub struct MyModelSelectionResult {
    model_selection_result: ModelSelectionResult,
}

impl From<MyModelSelectionResult> for Robj {
    fn from(my_model_selection_result: MyModelSelectionResult) -> Self {
        let mut list = List::from_values(&[
            r!(my_model_selection_result
                .model_selection_result
                .is_significant),
            r!(my_model_selection_result.model_selection_result.p_value),
        ]);

        list.set_names(&["is_significant", "p_value"])
            .expect("From<ModelSelectionResult> failed");

        list.into_robj()
    }
}

pub struct MyGainResult {
    gain_result: GainResult,
}

impl From<MyGainResult> for Robj {
    fn from(my_gain_result: MyGainResult) -> Self {
        match my_gain_result.gain_result {
            GainResult::FullGainResult(full_gain_result) => {
                let mut list = List::from_values(&[
                    r!(full_gain_result.start as i32),
                    r!(full_gain_result.stop as i32),
                    r!(full_gain_result.gain.to_vec()), // Convert ndarray to Vec
                ]);

                list.set_names(&["start", "stop", "gain"])
                    .expect("From<GainResult> failed");

                list.into_robj()
            }
            GainResult::ApproxGainResult(approx_gain_result) => {
                let mut list = List::from_values(&[
                    r!(approx_gain_result.start as i32),
                    r!(approx_gain_result.stop as i32),
                    r!(approx_gain_result.guess as i32),
                    r!(approx_gain_result.gain.to_vec()), // Convert ndarray to Vec
                    // For 2D arrays, convert to nested Vec or flatten
                    r!(approx_gain_result
                        .likelihoods
                        .outer_iter()
                        .map(|row| row.to_vec())
                        .collect::<Vec<Vec<f64>>>()),
                    r!(approx_gain_result.predictions.to_vec()),
                ]);

                list.set_names(&[
                    "start",
                    "stop",
                    "guess",
                    "gain",
                    "likelihoods",
                    "predictions",
                ])
                .expect("From<GainResult> failed");

                list.into_robj()
            }
        }
    }
}

pub struct MyOptimizerResult {
    optimizer_result: OptimizerResult,
}

impl From<MyOptimizerResult> for Robj {
    fn from(my_optimizer_result: MyOptimizerResult) -> Self {
        let gain_results: Vec<Robj> = my_optimizer_result
            .optimizer_result
            .gain_results
            .into_iter()
            .map(|gain_result| MyGainResult { gain_result }.into())
            .collect();

        let mut list = List::from_values(&[
            r!(my_optimizer_result.optimizer_result.start as i32),
            r!(my_optimizer_result.optimizer_result.stop as i32),
            r!(my_optimizer_result.optimizer_result.best_split as i32),
            r!(my_optimizer_result.optimizer_result.max_gain),
            r!(gain_results),
        ]);

        list.set_names(&["start", "stop", "best_split", "max_gain", "gain_results"])
            .expect("From<OptimizerResult> failed");

        list.into_robj()
    }
}

// Wrapper around BinarySegmentationResult to allow implementation of extendr_api::From
// e.g. https://stackoverflow.com/questions/25413201/how-do-i-implement-a-trait-i-dont-own-for-a-type-i-dont-own
pub struct MyBinarySegmentationResult {
    pub result: BinarySegmentationResult,
}

// Pass an object from rust to R (as a simple list).
// https://github.com/extendr/extendr/issues/308
impl From<MyBinarySegmentationResult> for Robj {
    fn from(my_result: MyBinarySegmentationResult) -> Self {
        // There exists no blanket implementation of From<Box<T>> where T: Into<Robj>.
        let left: Robj = match my_result.result.left {
            Some(boxed_tree) => MyBinarySegmentationResult {
                result: *boxed_tree,
            }
            .into(),
            None => ().into(),
        };

        let right: Robj = match my_result.result.right {
            Some(boxed_tree) => MyBinarySegmentationResult {
                result: *boxed_tree,
            }
            .into(),
            None => ().into(),
        };

        let optimizer_result: Robj = match my_result.result.optimizer_result.as_ref() {
            Some(optimizer_result) => MyOptimizerResult {
                optimizer_result: optimizer_result.clone(),
            }
            .into(),
            None => ().into(),
        };

        let segments: Robj = match my_result.result.segments {
            Some(optimizer_results) => {
                let robjs: Vec<Robj> = optimizer_results
                    .into_iter()
                    .map(|optimizer_result| MyOptimizerResult { optimizer_result }.into())
                    .collect();
                robjs.into()
            }
            None => ().into(),
        };

        let model_selection_result: Robj = MyModelSelectionResult {
            model_selection_result: my_result.result.model_selection_result.clone(),
        }
        .into();

        let mut list = List::from_values(&[
            r!(my_result.result.start as i32),
            r!(my_result.result.stop as i32),
            r!(my_result
                .result
                .optimizer_result
                .as_ref()
                .map(|result| result.best_split as i32)),
            r!(my_result
                .result
                .optimizer_result
                .as_ref()
                .map(|result| result.max_gain)),
            r!(my_result.result.model_selection_result.p_value),
            r!(my_result.result.model_selection_result.is_significant),
            r!(optimizer_result),
            r!(model_selection_result),
            r!(segments),
            r!(left),
            r!(right),
        ]);

        list.set_names(&[
            "start",
            "stop",
            "best_split",
            "max_gain",
            "p_value",
            "is_significant",
            "optimizer_result",
            "model_selection_result",
            "segments",
            "left",
            "right",
        ])
        .expect("From<Tree> failed");

        list.into_robj()
    }
}
