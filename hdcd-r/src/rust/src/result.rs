use extendr_api::prelude::*;
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

        // No From<Option<Vec>> for Robj: https://github.com/extendr/extendr/issues/313
        let gain: Robj = match my_result.result.gain {
            Some(gain) => gain.to_vec().into(),
            None => ().into(),
        };

        List::from_values(&[
            r!(my_result.result.start as i32),
            r!(my_result.result.stop as i32),
            r!(my_result.result.best_split.map(|u| u as i32)),
            r!(my_result.result.max_gain),
            r!(my_result.result.is_significant),
            r!(gain),
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
            "gain",
            "left",
            "right",
        ])
        .expect("From<Tree> failed")
    }
}
