```@meta
CurrentModule = Biwhitening
```

# Biwhitening

[Biwhitening](https://github.com/HolyLab/Biwhitening.jl) separates signal from noise in a
data matrix whose noise variance differs from entry to entry, as it does for counts. It
implements the method of [Landa, Zhang, and Kluger (2022)](@ref References):

1. Estimate the noise variance of each entry.
2. Scale the rows and columns of the data so that the noise has average variance one in
   every row and every column. The eigenvalues of pure noise then follow the
   Marchenko–Pastur law.
3. Count the singular values of the scaled matrix that lie above the largest one expected of
   noise. Their number estimates the rank of the signal.
4. Estimate the signal from those components.

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

The example uses simulated counts: a matrix of expected values of rank 5, whose rows and
columns differ in their totals, and Poisson noise.

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

Five singular values of the scaled matrix lie above [`noise_edge`](@ref), the largest
expected of noise:

```@example guide
round.(bw.S[1:8]; digits=1)', noise_edge(bw)
```

The result holds the scales `bw.a` and `bw.b`, every singular value `bw.S`, and the singular
vectors `bw.U` and `bw.V` of the components above the noise. The scaled matrix itself is
`Y ./ (bw.a .* bw.b')`.

## Checking the variance model

The rank is meaningful only if the variances given to `biwhiten` describe the noise. Two
numbers, both shown when a result is displayed, test this:

- [`noise_variance`](@ref) is the noise variance of the scaled matrix, estimated from its
  median eigenvalue. It should be close to one. A larger value means that the noise is
  larger than the model says, and the rank is then overestimated.
- [`ks_distance`](@ref) is the Kolmogorov–Smirnov distance between the eigenvalues and the
  Marchenko–Pastur law. Each component of the signal contributes one eigenvalue that the law
  does not describe, so the distance is at least the rank divided by the number of
  eigenvalues (here 5/400).

[`MarchenkoPastur`](@ref)`(bw)` is the law for the eigenvalues `bw.S .^ 2 ./ q`, where `q` is
the larger dimension of the matrix. Its density, `Biwhitening.pdf`, can be drawn over a
histogram of the eigenvalues.

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

The fit tries a grid of weights `β` of the quadratic term:

```@example guide
fit = fit_variance(Ynb; grid=0:0.1:1)
```

```@example guide
bwnb = biwhiten(Ynb, fit.model)
```

The fitted coefficients are close to the true ones (`b = 1`, `c = 0.25`), the noise variance
is one, and the rank is recovered.

Three points deserve attention when fitting:

- `fit.scan` lists every candidate. When the distance changes little over a range of `β`,
  the spectrum does not determine `β` within that range, and neither does it determine the
  rank more precisely than the ranks listed for those candidates.
- The distance of the chosen candidate is biased downward, because it was selected. To
  assess the model, fit it on part of the rows and compute `ks_distance` for the rest,
  biwhitened with the fitted model.
- With `grid=[0]`, the fit keeps the variance proportional to the expected value and
  estimates only the constant of proportionality.

For data that are not counts, pass a matrix of variance estimates as the second argument of
`biwhiten`.

## Estimating the signal

The singular values above the noise overestimate those of the signal, and the corresponding
singular vectors are tilted away from the signal's. A [`Shrinker`](@ref) replaces each
singular value by an estimate that accounts for both effects
([Gavish and Donoho, 2017](@ref References)); [`shrink`](@ref) returns the estimates.

```@example guide
round.([bw.S[1:5] shrink(bw, OperatorShrinker()) shrink(bw, FrobeniusShrinker())]; digits=1)
```

[`denoise`](@ref) assembles the components with shrunk singular values and multiplies by the
scales, which gives an estimate of the expected value of the data in its own units:

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
nonzero variance estimate receive a scale of zero and take no part in the result.

For some patterns of zeros no scaling exists: for example, a row whose only nonzero entries
lie in a few columns. `biwhiten` then throws an error that names the rows or columns
responsible. Either remove them, or pass `shrink=δ` with a small positive `δ`, which mixes
the variance estimates with those of a rank-one model so that the scaling always exists. The
[MatrixCovers documentation](https://holylab.github.io/MatrixCovers.jl/stable/noise/#noise-zeros)
describes the condition and the meaning of `δ`.

## Cost

`biwhiten` forms the dense Gram matrix of the shorter side of the data and computes all of
its eigenvalues. For an `m × n` matrix with `m ≤ n` this takes memory proportional to `m^2`
and time proportional to `m^2 * n + m^3`, whether or not the data are sparse.
`fit_variance` repeats this for every element of its grid.

## References

- M. Gavish and D. L. Donoho, "Optimal shrinkage of singular values," *IEEE Transactions on
  Information Theory* 63 (2017) 2137–2152.
  [doi:10.1109/TIT.2017.2653801](https://doi.org/10.1109/TIT.2017.2653801)
- B. Landa, T. T. C. K. Zhang, and Y. Kluger, "Biwhitening reveals the rank of a count
  matrix," *SIAM Journal on Mathematics of Data Science* 4 (2022) 1420–1446.
  [doi:10.1137/21M1456807](https://doi.org/10.1137/21M1456807)
