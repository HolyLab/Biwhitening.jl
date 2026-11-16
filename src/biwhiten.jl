"""
    Biwhitened{T}

The result of [`biwhiten`](@ref) for an `m × n` data matrix `Y`. Its fields are

- `a`, `b`: the row and column scales. The scaled matrix, [`scaled`](@ref)`(bw, Y)`, is
  `Y ./ (a .* b')`, with the rows and columns whose scale is zero set to zero.
- `S`: all singular values of the scaled matrix after removing those rows and columns, in
  decreasing order.
- `U`, `V`: the left and right singular vectors of the `k = rank(bw)` largest singular
  values, as the columns of an `m × k` and an `n × k` matrix.

`rank(bw)` is the number of singular values above [`noise_edge`](@ref).
[`noise_variance`](@ref) and [`ks_distance`](@ref) compare the spectrum with
[`MarchenkoPastur`](@ref)`(bw)`. [`shrink`](@ref), [`scores`](@ref), and [`denoise`](@ref)
estimate the signal.
"""
struct Biwhitened{T<:AbstractFloat}
    a::Vector{T}
    b::Vector{T}
    S::Vector{T}
    U::Matrix{T}
    V::Matrix{T}
end

Base.size(bw::Biwhitened) = (length(bw.a), length(bw.b))
Base.size(bw::Biwhitened, dim::Integer) = size(bw)[dim]

LinearAlgebra.rank(bw::Biwhitened) = size(bw.U, 2)

# The numbers of rows and columns with a nonzero scale, smaller first, and whether the rows
# are the smaller set.
function _shape(bw::Biwhitened)
    m, n = count(!iszero, bw.a), count(!iszero, bw.b)
    return min(m, n), max(m, n), m <= n
end

"""
    noise_edge(bw::Biwhitened)

The largest singular value of the scaled matrix expected of noise alone, `√m′ + √n′`, where
`m′` and `n′` are the numbers of rows and columns of the data with a nonzero scale.
`rank(bw)` is the number of singular values in `bw.S` that exceed it.
"""
function noise_edge(bw::Biwhitened{T}) where {T}
    p, q, _ = _shape(bw)
    return sqrt(T(p)) + sqrt(T(q))
end

"""
    MarchenkoPastur(bw::Biwhitened)

The distribution of the eigenvalues `bw.S .^ 2 ./ q` when the data consist only of noise with
the modeled variance. `q` is the larger of the numbers of rows and columns with a nonzero
scale.
"""
function MarchenkoPastur(bw::Biwhitened{T}) where {T}
    p, q, _ = _shape(bw)
    return MarchenkoPastur{T}(T(p) / T(q))
end

# The eigenvalues `S^2 / q` in increasing order.
function _eigenvalues(bw::Biwhitened)
    _, q, _ = _shape(bw)
    return reverse!(bw.S .^ 2 ./ q)
end

"""
    noise_variance(bw::Biwhitened)

The noise variance of the scaled matrix, estimated as the ratio of the median of its
eigenvalues to the median of the Marchenko–Pastur law. The median, unlike the mean, is
barely affected by the few eigenvalues that come from the signal. The estimate is close to
one when the variance estimates given to [`biwhiten`](@ref) have the right overall scale.
"""
noise_variance(bw::Biwhitened) = median(_eigenvalues(bw)) / median(MarchenkoPastur(bw))

"""
    ks_distance(bw::Biwhitened)

The Kolmogorov–Smirnov distance between the eigenvalues of the scaled matrix and the
Marchenko–Pastur law: the largest difference between the fraction of eigenvalues below a
point and the probability the law assigns below it. It is small when the variance estimates
given to [`biwhiten`](@ref) describe the noise and the rank of the signal is small compared
with the size of the matrix.
"""
ks_distance(bw::Biwhitened) = _ksdistance(_eigenvalues(bw), MarchenkoPastur(bw))

function Base.show(io::IO, bw::Biwhitened{T}) where {T}
    m, n = size(bw)
    print(io, "Biwhitened{", T, "}(", m, "×", n, ", rank ", rank(bw), ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", bw::Biwhitened{T}) where {T}
    m, n = size(bw)
    m′, n′ = count(!iszero, bw.a), count(!iszero, bw.b)
    print(io, "Biwhitened{", T, "} of a ", m, "×", n, " matrix")
    (m′, n′) == (m, n) || print(io, " (", m′, "×", n′, " without empty rows and columns)")
    σ2 = noise_variance(bw)
    k, kσ = rank(bw), count(>(sqrt(σ2) * noise_edge(bw)), bw.S)
    print(io, "\n  rank: ", k)
    k == kσ || print(io, " (", kσ, " if the noise has the estimated variance)")
    print(io, "\n  noise variance: ", round(σ2; sigdigits=4))
    print(io, "\n  Kolmogorov–Smirnov distance to the Marchenko–Pastur law: ", round(ks_distance(bw); sigdigits=3))
    return nothing
end

"""
    bw = biwhiten(Y, model::QuadraticVariance=QuadraticVariance(); kwargs...)
    bw = biwhiten(Y, V::AbstractMatrix; kwargs...)

Biwhiten the data matrix `Y`: find row scales `a` and column scales `b` such that the noise
of the scaled matrix `Y ./ (a .* b')` has average variance one in every row and every
column. Then compute all singular values of the scaled matrix, and the singular vectors of
those that exceed [`noise_edge`](@ref). The result is a [`Biwhitened`](@ref).

The noise is described either by a `model` that gives the variance of an entry as a function
of its expected value, or by a matrix `V` of variance estimates of the same size as `Y`. The
default model is the Poisson distribution, for which `V = Y`; this is the method of Landa,
Zhang, and Kluger (2022). Each estimate in `V` must be nonnegative, should be unbiased, and
must be computed from `Y` itself or from the measurements from which `Y` was derived.
[`fit_variance`](@ref) chooses a model from the data.

The scales are computed by `MatrixCovers.noise_cover`, to which all keywords are passed.
Rows and columns of `V` that are entirely zero receive a zero scale and are left out of the
spectrum. For some patterns of zeros in `V` no scaling exists, and `noise_cover` throws an
error that names the rows or columns responsible. Removing those rows or columns resolves
the error, as does passing the keyword `shrink=δ` with `δ > 0`. (This keyword is unrelated
to the function [`shrink`](@ref).)

`rank(bw)` assumes that the variance estimates have the right overall scale. Check that
[`noise_variance`](@ref)`(bw)` is close to one and that [`ks_distance`](@ref)`(bw)` is small
before relying on it. When the two differ, the display of `bw` also gives the number of
singular values above `sqrt(noise_variance(bw)) * noise_edge(bw)`, the count for noise of
the estimated variance.

# Examples

```jldoctest
julia> Y = [3 0 1 2; 0 2 5 1; 4 1 0 3];

julia> bw = biwhiten(Y);

julia> Vhat = Y ./ (bw.a .* bw.b') .^ 2;    # variance estimates for the scaled matrix

julia> round.(sum(Vhat; dims=2); digits=8)
3×1 Matrix{Float64}:
 4.0
 4.0
 4.0

julia> length(bw.S)
3
```
"""
function biwhiten(Y::AbstractMatrix{<:Real}, model::QuadraticVariance=QuadraticVariance(); kwargs...)
    Base.require_one_based_indexing(Y)
    T = float(eltype(Y))
    return _biwhiten(Y, _variances(model, Y, T), T; kwargs...)
end

function biwhiten(Y::AbstractMatrix{<:Real}, V::AbstractMatrix{<:Real}; kwargs...)
    Base.require_one_based_indexing(Y, V)
    size(Y) == size(V) ||
        throw(DimensionMismatch("the variance estimates must have the size of the data, $(size(Y)), but have size $(size(V))"))
    return _biwhiten(Y, V, float(promote_type(eltype(Y), eltype(V))); kwargs...)
end

"""
    Yh = scaled(bw::Biwhitened, Y)

The scaled matrix `Y ./ (bw.a .* bw.b')` of the data `Y` given to [`biwhiten`](@ref), with
the rows and columns whose scale is zero set to zero. This is the matrix that `biwhiten`
analyzes, and it can be passed to other algorithms, such as other factorization or
clustering methods. A sparse `Y` gives a sparse result.

# Examples

```jldoctest
julia> Y = [3 0 1 2; 0 2 5 1; 4 1 0 3; 0 0 0 0];

julia> bw = biwhiten(Y);

julia> Yh = scaled(bw, Y);

julia> iszero(Yh[4, :])                     # the empty row has scale zero
true

julia> all(isnan, (Y ./ (bw.a .* bw.b'))[4, :])
true
```
"""
function scaled(bw::Biwhitened, Y::AbstractMatrix)
    Base.require_one_based_indexing(Y)
    size(Y) == size(bw) ||
        throw(DimensionMismatch("the data must have the size of the biwhitened matrix, $(size(bw)), but have size $(size(Y))"))
    return Diagonal(_reciprocals(bw.a)) * Y * Diagonal(_reciprocals(bw.b))
end

# `1 ./ s`, except that zeros stay zero.
_reciprocals(s) = map(x -> iszero(x) ? x : inv(x), s)

function _biwhiten(Y, V, ::Type{T}; kwargs...) where {T}
    a, b, Yh, I, J = _normalize(Y, V, T; kwargs...)
    p, q = minmax(size(Yh)...)
    G = _gram(Yh)
    S = reverse!(sqrt.(max.(eigvals(G), zero(T))))
    k = count(>(sqrt(T(p)) + sqrt(T(q))), S)
    U, Vr = zeros(T, length(a), k), zeros(T, length(b), k)
    if k > 0
        # Singular vectors on the shorter side are eigenvectors of the Gram matrix; those on
        # the longer side follow from the scaled matrix.
        W = eigen(G, p-k+1:p).vectors[:, k:-1:1]
        if size(Yh, 1) <= size(Yh, 2)
            U[I, :] = W
            Vr[J, :] = (Yh' * W) ./ S[1:k]'
        else
            Vr[J, :] = W
            U[I, :] = (Yh * W) ./ S[1:k]'
        end
    end
    return Biwhitened{T}(a, b, S, U, Vr)
end

# The scales for the variance estimates `V`, the scaled matrix without the rows and columns
# whose scale is zero, and the indices of the rows and columns it keeps.
function _normalize(Y, V, ::Type{T}; kwargs...) where {T}
    a0, b0, _ = noise_cover(V; kwargs...)
    a, b = convert(Vector{T}, a0), convert(Vector{T}, b0)
    I, J = findall(!iszero, a), findall(!iszero, b)
    isempty(I) && throw(ArgumentError("every variance estimate is zero"))
    Ysub = (length(I), length(J)) == size(Y) ? Y : Y[I, J]
    Yh = Diagonal(inv.(a[I])) * Ysub * Diagonal(inv.(b[J]))
    return a, b, Yh, I, J
end

# The Gram matrix of the shorter side of `Yh`.
_gram(Yh::Matrix) = Symmetric(size(Yh, 1) <= size(Yh, 2) ? Yh * Yh' : Yh' * Yh)

# For other storage, such as sparse, the Gram matrix is accumulated from dense blocks taken
# along the longer side. The Gram matrix is dense even when `Yh` is sparse, and products of
# dense blocks are faster than a sparse product with a dense result.
function _gram(Yh::AbstractMatrix{T}) where {T}
    rowsshort = size(Yh, 1) <= size(Yh, 2)
    p, q = minmax(size(Yh)...)
    G = zeros(T, p, p)
    for blk in Iterators.partition(1:q, clamp(GRAMBLOCK ÷ p, 1, q))
        if rowsshort
            D = convert(Matrix{T}, Yh[:, blk])
            mul!(G, D, D', true, true)
        else
            D = convert(Matrix{T}, Yh[blk, :])
            mul!(G, D', D, true, true)
        end
    end
    return Symmetric(G)
end

# Number of elements in a dense block of `_gram`.
const GRAMBLOCK = 2^25
