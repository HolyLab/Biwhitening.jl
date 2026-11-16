# Biwhitening

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/Biwhitening.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/Biwhitening.jl/dev/)
[![Build Status](https://github.com/HolyLab/Biwhitening.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/Biwhitening.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/Biwhitening.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/Biwhitening.jl)

Biwhitening separates signal from noise in a data matrix whose noise variance differs from
entry to entry, such as a matrix of counts. When the noise has constant variance, its
eigenvalues follow the Marchenko–Pastur law, and components of the signal stand out above
the edge of that law. Unequal variances change the noise spectrum in a way that depends on
the unknown pattern of variances, so that no fixed threshold separates signal from noise.
Biwhitening restores the law: it scales the rows and columns so that the noise has average
variance one in each of them. The components above the edge can then be counted and
estimated. The method is that of Landa, Zhang, and Kluger,
["Biwhitening reveals the rank of a count matrix"](https://doi.org/10.1137/21M1456807)
(2022).

```julia
using Biwhitening

bw = biwhiten(Y)                # Y: a matrix of counts with Poisson noise
rank(bw)                        # number of components above the noise
noise_variance(bw)              # close to 1 if the Poisson model fits
ks_distance(bw)                 # distance of the spectrum from the Marchenko–Pastur law

fit = fit_variance(Y)           # choose a quadratic variance model for overdispersed counts
bw = biwhiten(Y, fit.model)

Yh = scaled(bw, Y)              # the scaled matrix, for use with other methods
Z = scores(bw, 1)               # coordinates of the rows on the components
Xhat = denoise(bw)              # estimate of the expected value of Y
```

The package provides

- `biwhiten`, for Poisson noise, for noise whose variance is a quadratic function of the
  expected value, or for a matrix of variance estimates supplied by the user;
- `fit_variance`, which chooses the variance function from the data;
- `MarchenkoPastur`, the reference distribution for the spectrum;
- shrinkage of the singular values (`shrink`, `scores`, `denoise`) by the rules of Gavish
  and Donoho (2017).

The row and column scales are computed by
[MatrixCovers.jl](https://github.com/HolyLab/MatrixCovers.jl). See the
[documentation](https://HolyLab.github.io/Biwhitening.jl/dev/) for a guide.
