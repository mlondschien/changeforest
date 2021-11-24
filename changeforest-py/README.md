# Classifier based non-parametric change point detection

Change point detection tries to identify times when the probability distribution of a
stochastic process or time series changes. Existing methods either assume a parametric
model for within-segment distributions or a based on ranks or distances, and thus fail
in scenarios with reasonably large dimensionality.

`changeforest` implements a classifier based algorithm that consistently estimates
change points without any parametric assumptions even in high-dimensional scenarios.
See [1] for details.

## References

[1] M. Londschien, S. Kovács and P. Bühlmann (2021), "Random Forests and other nonparametric classifiers for multivariate change point detection", working paper.