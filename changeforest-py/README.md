# Random Forests for Change Point Detection

Change point detection aims to identify structural breaks in the probability
distribution of a time series. Existing methods either assume a parametric model for
within-segment distributions or are based on ranks or distances and thus fail in
scenarios with a reasonably large dimensionality.

`changeforest` implements a classifier-based algorithm that consistently estimates
change points without any parametric assumptions, even in high-dimensional scenarios.
It uses the out-of-bag probability predictions of a random forest to construct a
pseudo-log-likelihood that gets optimized using a computationally feasible two-step
method.

See [1] for details.

## Installation

To install from `conda-forge` (recommended), run
```bash
conda install -c conda-forge changeforest
```

To install from `PyPI`, run
```bash
pip install changeforest
```

## Example

In the following example, we perform random forest-based change point detection on
a simulated dataset with `n=600` observations and covariance shifts at `t=200, 400`.

```python
In [1]: import numpy as np
   ...: 
   ...: Sigma = np.full((5, 5), 0.7)
   ...: np.fill_diagonal(Sigma, 1)
   ...: 
   ...: rng = np.random.default_rng(12)
   ...: X = np.concatenate(
   ...:     (
   ...:         rng.normal(0, 1, (200, 5)),
   ...:         rng.multivariate_normal(np.zeros(5), Sigma, 200),
   ...:         rng.normal(0, 1, (200, 5)),
   ...:     ),
   ...:     axis=0,
   ...: )
```

The simulated dataset `X` coincides with the _change in covariance_ (CIC) setup
described in [1]. Observations in the first and last segment are independently drawn
from a standard multivariate Gaussian distribution. Observations in the second segment
are i.i.d. normal with mean zero and unit variance, but with a covariance of ρ = 0.7
between coordinates. This is a challenging scenario.

```python
In [2]: from changeforest import changeforest
   ...: 
   ...: result = changeforest(X, "random_forest", "bs")
   ...: result
Out[2]: 
                    best_split max_gain p_value
(0, 600]                   412   19.603   0.005
 ¦--(0, 412]               201   62.981   0.005
 ¦   ¦--(0, 201]           194  -12.951    0.76
 ¦   °--(201, 412]         211   -9.211   0.545
 °--(412, 600]             418  -37.519   0.915

In [3]: result.split_points()
Out[3]: [201, 412]
```

`changeforest` correctly identifies the change point around `t=200` but is slightly
off at `t=412`. The `changeforest` function returns a `BinarySegmentationResult`.
We use its `plot` method to investigate the gain curves maximized by the change point estimates:

```
In [4]: result.plot().show()
```
<p align="center">
  <img src="../docs/py_cic_rf_binary_segmentation_result_plot.png" />
</p>
Change point estimates are marked in red.

For `method="random_forest"` and `method="knn"`, the `changeforest` algorithm uses a two-step approach to
find an optimizer of the gain. This fits a classifier for three split candidates
at the segment's 1/4, 1/2 and 3/4 quantiles, computes approximate gain curves using
the resulting pseudo-log-likelihoods and selects the overall optimizer as a second guess.
We can investigate the gain curves from the optimizer using the `plot` method of `OptimizerResult`.
The initial guesses are marked in blue.

```
In [5]: result.optimizer_result.plot().show()
```
<p align="center">
  <img src="../docs/py_cic_rf_optimizer_result_plot.png" />
</p>
 
One can observe that the approximate gain curves are piecewise linear, with maxima
around the true underlying change points.

The `BinarySegmentationResult` returned by `changeforest` is a tree-like object with attributes
`start`, `stop`, `best_split`, `max_gain`, `p_value`, `is_significant`, `optimizer_result`, `model_selection_result`, `left`, `right` and `segments`. 
These can be interesting to investigate the output of the algorithm further.

The `changeforest` algorithm can be tuned with hyperparameters. See
[here](https://github.com/mlondschien/changeforest/blob/287ac0f10728518d6a00bf698a4d5834ae98715d/src/control.rs#L3-L30)
for their descriptions and default values. In Python, the parameters can
be specified with the [`Control` class](https://github.com/mlondschien/changeforest/blob/b33533fe0ddf64c1ea60d0d2203e55b117811667/changeforest-py/changeforest/control.py#L1-L26),
which can be passed to `changeforest`. The following will build random forests with
20 trees:

```python
In [6]: from changeforest import Control
   ...: changeforest(X, "random_forest", "bs", Control(random_forest_n_estimators=20))
Out[6]: 
                            best_split max_gain p_value
(0, 600]                           592  -11.786    0.01
 ¦--(0, 592]                       121    -6.26   0.015
 ¦   ¦--(0, 121]                    13  -14.219   0.615
 ¦   °--(121, 592]                 416   21.272   0.005
 ¦       ¦--(121, 416]             201   37.157   0.005
 ¦       ¦   ¦--(121, 201]         192   -17.54    0.65
 ¦       ¦   °--(201, 416]         207   -6.701    0.74
 ¦       °--(416, 592]             584  -44.054   0.935
 °--(592, 600]     
```

The `changeforest` algorithm still detects change points around `t=200, 400` but also
returns two false positives.

Due to the nature of the change, `method="change_in_mean"` is unable to detect any
change points at all:
```python
In [7]: changeforest(X, "change_in_mean", "bs")
Out[7]: 
          best_split max_gain p_value
(0, 600]         589    8.318 
```

## References

[1] M. Londschien, S. Kovács and P. Bühlmann (2022), "Random Forests for Change Point Detection", working paper.
