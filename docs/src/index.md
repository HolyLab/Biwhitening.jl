```@meta
CurrentModule = Biwhitening
```

# Biwhitening

[Biwhitening](https://github.com/HolyLab/Biwhitening.jl) separates signal from noise in a
data matrix whose noise variance differs from entry to entry, as it does for counts. When
the noise has constant variance, its eigenvalues follow the Marchenko–Pastur law, and
components of the signal stand out above the edge of that law. Unequal variances change the
noise spectrum in a way that depends on the unknown pattern of variances, so that no fixed
threshold separates signal from noise. The method of
[Landa, Zhang, and Kluger (2022)](@ref References) restores the law by rescaling the data:

1. Estimate the noise variance of each entry.
2. Scale the rows and columns of the data so that the noise has average variance one in
   every row and every column.
3. Count the singular values of the scaled matrix that exceed the largest singular value
   expected of noise. Their number estimates the rank of the signal.
4. Estimate the signal from those components.

Unlike a variance-stabilizing transformation, the rescaling in step 2 leaves the rank of the
signal unchanged; see
[Comparison with variance-stabilizing transformations](@ref).

The scaling of step 2 is computed by
[MatrixCovers.jl](https://github.com/HolyLab/MatrixCovers.jl), whose
[documentation](https://holylab.github.io/MatrixCovers.jl/stable/noise/) discusses how to
estimate variances for several kinds of data.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/HolyLab/Biwhitening.jl")
```

## A first example

The example simulates counts with Poisson noise around a matrix `X` of expected values. `X`
has rank 5, and its rows and columns differ widely in scale, so the noise variance differs
from entry to entry.

```@example guide
using Biwhitening, LinearAlgebra, Random

function poisson(rng, λ)        # one Poisson variable with mean λ
    k, t = 0, randexp(rng)
    while t < λ
        k += 1
        t += randexp(rng)
    end
    return k
end

rng = Xoshiro(1)
m, n, r = 400, 1200, 5
X = (rand(rng, m, r) * rand(rng, r, n)) .* (0.2 .+ 3 .* rand(rng, m)) .* (0.2 .+ 3 .* rand(rng, n))'
Y = [poisson(rng, x) for x in X]
nothing # hide
```

[`biwhiten`](@ref) assumes Poisson noise unless told otherwise:

```@example guide
bw = biwhiten(Y)
```

Five singular values of the scaled matrix exceed [`noise_edge`](@ref), the largest singular
value expected of noise:

```@example guide
round.(bw.S[1:8]; digits=1)', noise_edge(bw)
```

The result holds the scales `bw.a` and `bw.b`, every singular value `bw.S`, and the singular
vectors `bw.U` and `bw.V` of the components above the noise.

[`scaled`](@ref)`(bw, Y)` returns the scaled matrix itself.

## Checking the variance model

The rank is meaningful only if the variances given to `biwhiten` describe the noise. Two
numbers, both shown when a result is displayed, test this:

- [`noise_variance`](@ref) is the noise variance of the scaled matrix, estimated from its
  median eigenvalue. It should be close to one. A larger value means that the noise exceeds
  what the model predicts, and the rank is then overestimated. When the estimate changes
  the count, the display adds the rank for noise of the estimated variance.
- [`ks_distance`](@ref) is the Kolmogorov–Smirnov distance between the eigenvalues and the
  Marchenko–Pastur law. Each component of the signal contributes one eigenvalue that the law
  does not describe, so the distance is at least the rank divided by the number of
  eigenvalues (here 5/400).

`d = `[`MarchenkoPastur`](@ref)`(bw)` is the law for the eigenvalues `bw.S .^ 2 ./ q`, where
`q` is the larger dimension of the matrix. Its probability density, the function
[`Biwhitening.pdf`](@ref)`(d, x)`, can be drawn over a histogram of the eigenvalues.

## Choosing the variance model

Counts from sequencing are often more variable than the Poisson distribution allows. A
[`QuadraticVariance`](@ref) describes noise whose variance is a quadratic function of the
expected value, `a + b*x + c*x^2`, and [`fit_variance`](@ref) chooses `b` and `c` by
comparing the spectrum with the Marchenko–Pastur law.

The counts below have a negative binomial distribution with variance `x + x^2 / 4`. Under the
Poisson model the noise variance is far from one and the rank is wrong:

```@example guide
gamma4(rng) = sum(randexp(rng) for _ in 1:4) / 4     # gamma variable, shape 4, mean 1
Ynb = [poisson(rng, x * gamma4(rng)) for x in X]
biwhiten(Ynb)
```

The fit tries a grid of weights `β` for the quadratic term and keeps the one whose spectrum
is closest to the law:

```@example guide
fit = fit_variance(Ynb; grid=0:0.1:1)
```

```@example guide
bwnb = biwhiten(Ynb, fit.model)
```

The fitted coefficients are close to the true ones (`b = 1`, `c = 0.25`), the noise variance
is close to one, and the rank is recovered.

Three caveats apply to the fit:

- `fit.scan` lists every candidate. When the distance changes little over a range of `β`,
  the spectrum does not determine `β` within that range. The rank is then uncertain to the
  extent that the candidates in that range disagree about it.
- Because the chosen candidate was selected for its small distance, that distance is biased
  downward. To assess the model without this bias, fit it on some of the rows, biwhiten the
  remaining rows with the fitted model, and compute their `ks_distance`.
- With `grid=[0]`, the fit keeps the variance proportional to the expected value and
  estimates only the constant of proportionality.

For data that are not counts, pass a matrix of variance estimates, of the same size as the
data, as the second argument of `biwhiten`.

## Estimating the signal

The singular values above the noise overestimate those of the signal, and the corresponding
singular vectors are tilted away from the signal's. A [`Shrinker`](@ref) replaces each
singular value by an estimate that accounts for both effects
([Gavish and Donoho, 2017](@ref References)); [`shrink`](@ref) returns the estimates.

```@example guide
round.([bw.S[1:5] shrink(bw, OperatorShrinker()) shrink(bw, FrobeniusShrinker())]; digits=1)
```

[`denoise`](@ref) rebuilds the matrix from the components with shrunk singular values and
undoes the scaling. The result estimates the expected value of the data; below, its relative
error is compared with that of the data themselves:

```@example guide
Xhat = denoise(bw)
norm(Xhat - X) / norm(X), norm(Y - X) / norm(X)
```

[`scores`](@ref) gives coordinates of the rows or columns on the components, for
visualization or clustering:

```@example guide
Z = scores(bw, 1)       # one row for each row of Y, one column for each component
size(Z)
```

With the default [`HardThreshold`](@ref) these are the principal component scores of the
scaled matrix, and distances between rows are in units of the noise: on a component well
above the edge, each coordinate carries noise of about unit variance.

The rules differ in what they estimate best:

| rule | best estimate of |
|:--|:--|
| [`FrobeniusShrinker`](@ref) | the signal, by the sum of squared errors of its entries |
| [`OperatorShrinker`](@ref) | each singular value of the signal |
| [`NuclearShrinker`](@ref) | the signal, by the sum of the singular values of the error |
| [`GramShrinker`](@ref) | inner products between rows, or between columns, of the signal |
| [`HardThreshold`](@ref) | none; the singular values are unchanged |

## Zeros, empty rows, and sparse data

`biwhiten` accepts sparse matrices and matrices of integers. Rows and columns without any
nonzero variance estimate receive a scale of zero and are left out of the spectrum.

For some patterns of zeros no scaling exists: for example, a row whose nonzero entries lie in
too few columns. `biwhiten` then throws an error that names the rows or columns responsible.
Either remove them, or pass the keyword `shrink=δ` with a small positive `δ`. (This keyword
is unrelated to the function [`shrink`](@ref).) It mixes the variance estimates with those of
a rank-one model, so that the scaling always exists. The
[MatrixCovers documentation](https://holylab.github.io/MatrixCovers.jl/stable/noise/#noise-zeros)
describes the condition and the meaning of `δ`.

## Comparison with variance-stabilizing transformations

A variance-stabilizing transformation equalizes noise variances in a different way: it
applies a nonlinear function to each entry, such as the square root for Poisson counts or
`log(1 + y)`, so that the noise variance is approximately the same in every entry.
Biwhitening equalizes only the average variance of each row and column. In exchange, it
preserves linearity, on which matrix factorization depends.

Factorization methods such as principal component analysis represent the data as a sum of
components, and the rank of the signal is the number of components needed. Biwhitening
divides each entry by a constant, so the expected value of the scaled matrix,
`X ./ (a .* b')`, has the same rank as the expected value `X` of the data
([Landa, Zhang, and Kluger, 2022](@ref References), eq. 2.3), and counts that add before
scaling still add after it. A nonlinear transformation `f` preserves neither property:
`f.(X)` generally has higher rank than `X`, and the expected value of `f.(Y)` is not
`f.(X)`. The components found in transformed data therefore need not correspond to those of
the signal, and their number need not equal its rank.

Both approaches need a model of the noise variance: the square root stabilizes Poisson noise,
for example, but not overdispersed noise. Variance-stabilizing transformations are also
approximate, and least accurate for small expected values, which predominate in sparse
counts. Biwhitening rests instead on an asymptotic theorem, which assumes that the noise is
independent from entry to entry (Landa, Zhang, and Kluger, Theorem 2.2 and
Proposition 4.1). Finally, because the scales weight rows and columns unequally, the singular
vectors of the scaled matrix differ from those of the unscaled signal. [`denoise`](@ref)
undoes the scaling, but [`scores`](@ref) are coordinates in the scaled matrix.

## Cost

For an `m × n` matrix with `m ≤ n`, `biwhiten` forms the dense `m × m` Gram matrix of the
scaled data and computes all of its eigenvalues. This takes memory proportional to `m^2` and
time proportional to `m^2 * n + m^3`, whether or not the data are sparse. `fit_variance`
repeats this for every element of its grid.

## References

- M. Gavish and D. L. Donoho, "Optimal shrinkage of singular values," *IEEE Transactions on
  Information Theory* 63 (2017) 2137–2152.
  [doi:10.1109/TIT.2017.2653801](https://doi.org/10.1109/TIT.2017.2653801)
- B. Landa, T. T. C. K. Zhang, and Y. Kluger, "Biwhitening reveals the rank of a count
  matrix," *SIAM Journal on Mathematics of Data Science* 4 (2022) 1420–1446.
  [doi:10.1137/21M1456807](https://doi.org/10.1137/21M1456807)
