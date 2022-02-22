
# Changelog

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

  This increases estimation performance for classifier based methods, especially if used
  with standard binary segmentation, i.e. for `changeforst_bs` and `changeforest_knn`.

## 0.1.1 - (2021-11-25)

**Other changes:**

- Added license file for compatability with conda-forge.
