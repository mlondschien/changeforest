# Changelog

## 0.2.1 - (2021-12-13)

**Bug fixes:**

- (Python only) Parameters will now be correctly passed to `changeforest` via `Control` even
  if they have an incorrect data type.
- Fixed a bug where SBS would panic in cases with very small minimal segments lengths
  due to rounding.
- Fixed a bug in model selection that resulted in a higher type I error.


## 0.2.0 - (2021-12-10)

**New feature**:

- The `TwoStepSearch` now uses three initial guesses, the 0.25, 0.5 and 0.75 quantiles
  of the segment, for the first step. The the best split corresponding to the highest
  maximal gain from the three guesses is used in the second step. The permutation test
  used for model selection has also been adjusted to be consistent.

  This increases estimation performance for classifier based methods, especially if used
  with standard binary segmentation, i.e. for `changeforst_bs` and `changeforest_knn`.

## 0.1.1 - (2021-11-25)

**Other changes**:

- Added license file for compatability with conda-forge.
