
# Changelog

## 1.1.3 - (2024-01-06)

**Other changes:**

- Update the `extendr` architecture for the package build. Thanks @JosiahParry!

## 1.1.2 - (2024-01-02)

**Other changes:**

- Upgraded `extendr-api` and `ndarray` dependencies in R package.

## 1.1.1 - (2023-10-03)

**Other changes:**

- Upgraded `pyo3` dependency in Python package.

## 1.1.0 - (2023-08-01)

**New features**:

- New argument `forbidden_segments` (list or vector of 2-tuple) or `None` to `Control`. If not `None`, `changeforest` will not split on split points contained in segments `(a, b]` in `forbidden_segments` (rust and Python only). Thanks @enzbus!

## 1.0.1 - (2022-06-01)

**Bug fixes:**

- Python macos images are now again correctly built on GitHub runners.

## 1.0.0 - (2022-05-30)

First major release. There have been no changes since the last release. The manuscript is to be published in JMLR.

## 0.7.2 - (2022-05-09)

**Bug fixes:**

- Fixed bugs when plotting results created with `method="change_in_mean"` or `segmentation="sbs"` or `"wbs"` (Python).

## 0.7.1 - (2022-05-02)

**Bug fixes:**

- Fixed a bug resulting in no tick-labels being shown on the x-axis when plotting a `BinarySegmentationResult`.

## 0.7.0 - (2022-04-08)

**New features**:

- New plotting methods `BinarySegmentationResult.plot` and `OptimizerResult.plot` (Python).
- New plotting methods `plot.binary_segmentation_result` and `plot.binary_segmentation_result` (R).
- Expanded documentation (R).
- The `changeforest` function now has default values `method="random_forest"` and `segmentation="bs"` (R).

## 0.6.1 - (2022-04-06)

**Bug fixes:**

- Fixed a bug in the Python package when passing `random_forest_max_features='sqrt'` to `Control`.

## 0.6.0 - (2022-03-17)

**Breaking changes:**

- The default value for `model_selection_n_permutations` is now 199.
- The default value for `model_selection_alpha` is now 0.02.
- The default value for `minimal_gain_to_split`, use in the `change_in_mean` setup, is now `log(n) * (d + 1)`, motivated by the BIC and [1].
- The value for `minimal_gain_to_split` no longer gets automatically multiplied by `n`.

[1] Yao, Y.-C. (1988). Estimating the number of change-points via Schwarz’ criterion. Statist. Probab. Lett. 6 181–189. MR0919373

## 0.5.1 - (2022-03-16)

**Bug fixes:**

- The pseudo-permutation-test now correctly skips the first and last `minimal_segment_length * n` observations when calculating the permuted maximal gains.

**Other changes:**

- The first three elements of the `result.optimizer_result.gain_results` returned by the two-step search are no longer sorted by their maximal gain.

## 0.5.0 - (2022-03-15)

**Breaking changes:**

- The parameters `random_forest_mtry` and `random_forest_n_trees` of `Control` have been renamed to `random_forest_max_features` and `random_forest_n_estimators`.
- The default value for `random_forest_max_features` now is `floor(sqrt(d))`.

**New features:**

- The parameter `random_forest_max_features` now can be supplied with a fraction `0 < f < 1`, an integer `i>=1`, `None` (Python, Rust) / `NULL` (R) and `"sqrt"`. Then, for each split, repsectively `floor(f d)`, `i`, `d` or `floor(sqrt(d))` features are considered.

**Other changes:**

- Bump `biosphere` dependency to 0.3.0

## 0.4.4 - (2022-02-22)

**Other changes:**

- Bump `biosphere` dependency to 0.2.2.

## 0.4.3 - (2021-01-29)

**Other changes:**

- The default value for `Control.minimal_gain_to_split` is now `log(n_samples) * n_features / n_samples`,
motivated by the Bayesian information criterion (BIC). 

## 0.4.2 - (2021-01-21)

**Other changes:**

- The R-package now makes use of the latest version of `libR-sys`, enabling compilation for Apple silicon on `conda-forge` (#86).

**Bug fixes:**

- Fixed a bug where passing `Control()` to `changeforest` in the Python package overwrote the default value for `random_forest_max_depth` to `None`. Default values for `Control` in the python package are now `"default"` (#87).

## 0.4.1 - (2021-01-13)

**Bug fixes:**

- Upgrade `biosphere` to `0.2.1` fixing a bug in `RandomForest` (#84).

**Other changes:**

- New parameter `model_selection_n_permutations` (#85).

## 0.4.0 - (2021-01-11)

**New features:**

- `changeforest` now uses random forests from [`biosphere`](https://github.com/mlondschien/biosphere).
  This should be faster than `smartcore` used previously and supports parallelization (#82).

## 0.3.0 - (2021-12-15)

**New features:**

- Implemented trait `Display` for `BinarySegmentationResult`. In Python `str(result)` now prints a pretty display (#77).

**Other changes:**

- The `TwoStepSearch` algorithm now only uses valid guesses from `split_candidates` (#76).

**Bug fixes:**

- (R only) The R6 class `Control` now gets correctly exported (#79).

## 0.2.1 - (2021-12-13)

**Bug fixes:**

- (Python only) Parameters will now be correctly passed to `changeforest` via `Control` even
  if they have an incorrect data type (#67).
- Fixed a bug where SBS would panic in cases with very small minimal segments lengths
  due to rounding (#70).
- Fixed a bug in model selection that resulted in a higher type I error (#71).


## 0.2.0 - (2021-12-10)

**New features:**

- The `TwoStepSearch` now uses three initial guesses, the 0.25, 0.5 and 0.75 quantiles
  of the segment, for the first step. The the best split corresponding to the highest
  maximal gain from the three guesses is used in the second step. The permutation test
  used for model selection has also been adjusted to be consistent (#65).

  This increases estimation performance for classifier-based methods, especially if used
  with standard binary segmentation, i.e. for `changeforst_bs` and `changeforest_knn`.

## 0.1.1 - (2021-11-25)

**Other changes:**

- Added license file for compatability with conda-forge.
