use extendr_api::prelude::*;
//use extendr_api::robj_ndarray::TryFrom;
use hdcd::gain::GainResult;
use hdcd::optimizer::OptimizerResult;
use hdcd::BinarySegmentationResult;

pub struct MyGainResult {
    gain_result: GainResult,
}

impl From<MyGainResult> for Robj {
    fn from(my_gain_result: MyGainResult) -> Self {
        match my_gain_result.gain_result {
            GainResult::FullGainResult(full_gain_result) => List::from_values(&[
                r!(full_gain_result.start as i32),
                r!(full_gain_result.stop as i32),
                r!(Robj::try_from(&full_gain_result.gain).unwrap()),
            ])
            .into_robj()
            .set_names(&["start", "stop", "gain"])
            .expect("From<GainResult> failed"),
            GainResult::ApproxGainResult(approx_gain_result) => List::from_values(&[
                r!(approx_gain_result.start as i32),
                r!(approx_gain_result.stop as i32),
                r!(approx_gain_result.guess as i32),
                r!(Robj::try_from(&approx_gain_result.gain).unwrap()),
                r!(Robj::try_from(&approx_gain_result.likelihoods).unwrap()),
                r!(Robj::try_from(&approx_gain_result.predictions).unwrap()),
            ])
            .into_robj()
            .set_names(&[
                "start",
                "stop",
                "guess",
                "gain",
                "likelihoods",
                "predictions",
            ])
            .expect("From<GainResult> failed"),
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

        List::from_values(&[
            r!(my_optimizer_result.optimizer_result.start as i32),
            r!(my_optimizer_result.optimizer_result.stop as i32),
            r!(my_optimizer_result.optimizer_result.best_split as i32),
            r!(my_optimizer_result.optimizer_result.max_gain),
            r!(gain_results),
        ])
        .into_robj()
        .set_names(&["start", "stop", "best_split", "max_gain", "gain_results"])
        .expect("From<OptimizerResult> failed")
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

        let gain_results: Robj = match my_result.result.gain_results {
            Some(results) => {
                let robjs: Vec<Robj> = results
                    .into_iter()
                    .map(|gain_result| MyGainResult { gain_result }.into())
                    .collect();
                robjs.into()
            }
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

        List::from_values(&[
            r!(my_result.result.start as i32),
            r!(my_result.result.stop as i32),
            r!(my_result.result.best_split.map(|u| u as i32)),
            r!(my_result.result.max_gain),
            r!(my_result.result.is_significant),
            r!(gain_results),
            r!(segments),
            r!(left),
            r!(right),
        ])
        .into_robj()
        .set_names(&[
            "start",
            "stop",
            "best_split",
            "max_gain",
            "is_significant",
            "gain_results",
            "segments",
            "left",
            "right",
        ])
        .expect("From<Tree> failed")
    }
}
