# Random Forests for Change Point Detection

Change point detection aims to identify structural breaks in the probability
distribution of a time series. Existing methods either assume a parametric model for
within-segment distributions or a based on ranks or distances, and thus fail in
scenarios with reasonably large dimensionality.

`changeforest` implements a classifier-based algorithm that consistently estimates
change points without any parametric assumptions even in high-dimensional scenarios.
It uses the out-of-bag probability predictions of a random forest to construct a
pseudo-log-likelihood that gets optimized using a computationally feasible two-step
method.

See [1] for details.


`changeforest` is available as rust crate, a Python package (on
[`PyPI`](https://pypi.org/project/changeforest/) and
[`conda-forge`](https://anaconda.org/conda-forge/changeforest))
and as an R package (on [`conda-forge`](https://anaconda.org/conda-forge/r-changeforest)
, linux and MacOS only). See below for their respective user guides.

## Python

### Installation

To install from `conda-forge` (recommended), simply run
```bash
conda install -c conda-forge changeforest
```

To install from `PyPI`, run
```bash
pip install changeforest
```

### Example

In the following example we perform random forest-based change point detection on
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

`changeforest` correctly identifies the change point around `t=200`, but is slightly
off at `t=412`. The `changeforest` function returns a `BinarySegmentationResult`.
We use its `plot` method to investigate the gain curves maximized by the change point estimates:

```
result.plot().show()
```
<p align="center">
  <img src="docs/py_cic_rf_binary_segmentation_result_plot.png" />
</p>
Change point estimates are marked in red.

For `method="random_forest"` (and `method="knn"`), the `changeforest` algorithm uses a two-step approach to
find an optimizer of the gain. This fits a classifier for three split candidates
at the 1/4, 1/2 and 3/4 quantiles of the segment, computes approximate gain curves using
the resulting pseudo-log-likelihoods and selects the overall optimizer as a second guess.
We can investigate the gain curves from the optimizer using the `plot` method of `OptimizerResult`.

```
result.optimizer_result.plot().show()
```
<p align="center">
  <img src="docs/py_cic_rf_optimizer_result_plot.png" />
</p>
 
One can clearly observe that the approximate gain curves are piecewise linear, with maxima
at the true underlying change points.

The `BinarySegmentationResult` returned by `changeforest` is a tree-like object with attributes
`start`, `stop`, `best_split`, `max_gain`, `p_value`, `is_significant`, `optimizer_result`, `model_selection_result`, `left`, `right` and `segments`. 
These can be interesting to further investigate the output of the algorithm.

The `changeforest` algorithm can be tuned with hyperparameters. See
[here](https://github.com/mlondschien/changeforest/blob/287ac0f10728518d6a00bf698a4d5834ae98715d/src/control.rs#L3-L30)
for their descriptions and default values. In Python, the parameters can
be specified with the [`Control` class](https://github.com/mlondschien/changeforest/blob/b33533fe0ddf64c1ea60d0d2203e55b117811667/changeforest-py/changeforest/control.py#L1-L26)
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

The `changeforest` algorithm still detects change points around `t=200, 400`, but also
returns two false-positives.

Due to the nature of the change, `method="change_in_mean"` is unable to detect any
change points at all:
```python
In [7]: changeforest(X, "change_in_mean", "bs")
Out[7]: 
          best_split max_gain p_value
(0, 600]         589    8.318 
```

## R

To install from `conda-forge`, simply run

```bash
conda install -c conda-forge r-changeforest
```

### Example

The following example performs random forest based change point detection on the iris
dataset. This includes three classes _setosa_, _versicolor_ and _virginica_ with 50
observations each. We interpret this as a simulated time series with change points at
`t = 50, 100`.

```R
> library(changeforest)
> library(datasets)
> data(iris)
> X <- as.matrix(iris[, 1:4])
> changeforest(X, "random_forest", "bs")
                name best_split  max_gain p_value is_significant
1 (0, 150]                   50 96.173664    0.01           TRUE
2  ¦--(0, 50]                34 -5.262184    1.00          FALSE
3  °--(50, 150]             100 51.557473    0.01           TRUE
4      ¦--(50, 100]          80 -3.068934    1.00          FALSE
5      °--(100, 150]        134 -2.063508    1.00          FALSE
```

`changeforest` also implements methods `change_in_mean` and `knn`. While `random_forest`
and `knn` implement the `TwoStepSearch` optimizer as described in [1], for
`change_in_mean` the optimizer `GridSearch` is used. Both `random_forest` and `knn`
perform model selection via a pseudo-permutation test (see [1]). For `change_in_mean`
split candidates are kept whenever `max_gain > control.minimal_gain_to_split`.

The iris dataset allows for rather simple classification due to large mean shifts between classes. As a
result, both `change_in_mean` and `knn` also correctly identify die true change points.

```R
> result <- changeforest(X, "change_in_mean", "bs")
> result$split_points()
[1] [50, 100]
> result <- changeforest(X, "knn", "bs")
> result$split_points()
[1] [50, 100]
```

`changeforest` returns a tree-like object with attributes `start`, `stop`, `best_split`, `max_gain`, `p_value`, `is_significant`, `optimizer_result`, `model_selection_result`, `left`, `right` and `segments`. These can be interesting to further investigate the output of the algorithm. Here we
plot the approximated gain curves of the first three segments:
```R
> library(ggplot2)
> result <- changeforest(X, "random_forest", "bs")
> data = data.frame(
        t=1:150,
        gain=result$optimizer_result$gain_results[[3]]$gain,
        gain_left=c(result$left$optimizer_result$gain_results[[3]]$gain, rep(NA, 100)),
        gain_right=c(rep(NA, 50), result$right$optimizer_result$gain_results[[3]]$gain)
)

> ggplot(data=data) +
        geom_line(aes(x=t, y=gain), color="blue") + 
        geom_line(aes(x=t, y=gain_left), color="orange") + 
        geom_line(aes(x=t, y=gain_right), color="green") +
        labs(y = 'gain') + theme(legend.position="bottom")
```

<p align="center">
  <img src="docs/r-iris-approx-gains.png" />
</p>

One can clearly observe that the approximate gain curves are piecewise linear, with maxima
at the true underlying change points.

The `changeforest` algorithm can be tuned with hyperparameters. See [here](https://github.com/mlondschien/changeforest/blob/b33533fe0ddf64c1ea60d0d2203e55b117811667/src/control.rs#L3-L39)
for their descriptions and default values. In R, the parameters can
be specified with the [`Control` class](https://github.com/mlondschien/changeforest/blob/main/changeforest-r/R/control.R)
which can be passed to `changeforest`. The following
will build random forests with very few trees:

```R
> changeforest(X, "random_forest", "bs", Control(random_forest_n_estimators=10))
... TODO
```

## References

[1] M. Londschien, S. Kovács and P. Bühlmann (2022), "Random Forests for Change Point Detection", working paper.
