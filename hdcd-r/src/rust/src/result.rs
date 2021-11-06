use extendr_api::prelude::*;
//use extendr_api::robj_ndarray::TryFrom;
use hdcd::gain::GainResult;
use hdcd::BinarySegmentationResult;

// Wrapper around BinarySegmentationResult to allow implementation of extendr_api::From
// e.g. https://stackoverflow.com/questions/25413201/how-do-i-implement-a-trait-i-dont-own-for-a-type-i-dont-own
pub struct MyBinarySegmentationResult {
    pub result: BinarySegmentationResult,
}

// https://github.com/extendr/extendr/issues/308
impl From<MyBinarySegmentationResult> for Robj {
    fn from(my_result: MyBinarySegmentationResult) -> Self {
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
                // value of type `extendr_api::Robj` cannot be built from
                // `std::iter::Iterator<Item=extendr_api::Robj>`
                let mut robjs = vec![];

                for result in results {
                    let robj: Robj = match result {
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
                    };
                    robjs.push(robj);
                }
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
            "left",
            "right",
        ])
        .expect("From<Tree> failed")
    }
}
