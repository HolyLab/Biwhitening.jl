"""
    Shrinker

A rule for estimating the singular values of the signal from those singular values of the
scaled matrix that exceed [`noise_edge`](@ref). The rules are [`HardThreshold`](@ref),
[`FrobeniusShrinker`](@ref), [`OperatorShrinker`](@ref), [`NuclearShrinker`](@ref), and
[`GramShrinker`](@ref); [`shrink`](@ref) applies one.

Noise inflates the singular values of the signal and tilts its singular vectors. Let `q` be
the larger dimension of the scaled matrix. In the limit of a large matrix, a component of
the signal with singular value `x√q` appears in the scaled matrix with a larger singular
value `y√q`. The angles between its observed singular vectors and those of the signal have
cosines `c₁` for the left singular vector (in `bw.U`) and `c₂` for the right (in `bw.V`). `x`, `c₁`, and `c₂` are functions of `y` and of the aspect ratio of the matrix, and
each rule is a function of them (Gavish and Donoho, 2017).
"""
abstract type Shrinker end

"""
    HardThreshold()

The [`Shrinker`](@ref) that leaves the singular values above the noise unchanged.
"""
struct HardThreshold <: Shrinker end

"""
    FrobeniusShrinker()

The [`Shrinker`](@ref) `x * c₁ * c₂`, for which the estimate of the signal has the smallest
expected squared error summed over its entries.
"""
struct FrobeniusShrinker <: Shrinker end

"""
    OperatorShrinker()

The [`Shrinker`](@ref) `x`, which estimates the singular value of the signal itself. The
resulting estimate of the signal has the smallest error in the operator norm.
"""
struct OperatorShrinker <: Shrinker end

"""
    NuclearShrinker()

The [`Shrinker`](@ref) for which the estimate of the signal has the smallest error in the
nuclear norm, `x * (c₁ * c₂ - sqrt((1 - c₁^2) * (1 - c₂^2)))` where that is positive and
zero elsewhere.
"""
struct NuclearShrinker <: Shrinker end

"""
    GramShrinker(dim)

The [`Shrinker`](@ref) `x * c₁` (`dim = 1`) or `x * c₂` (`dim = 2`). With `Z` the
[`scores`](@ref) of the rows (`dim = 1`) or columns (`dim = 2`) under this rule, `Z * Z'`
has the smallest expected squared error as an estimate of the matrix of inner products
between the rows, or columns, of the signal. It applies to `scores` and not to
[`denoise`](@ref).
"""
struct GramShrinker <: Shrinker
    dim::Int

    function GramShrinker(dim::Integer)
        dim in (1, 2) || throw(ArgumentError("`dim` must be 1 or 2, got $dim"))
        return new(dim)
    end
end

# For a singular value `y√q` above the noise in a matrix of aspect ratio `γ ≤ 1`: the
# strength `x` of the signal and the cosines between its singular vectors and the observed
# ones on the shorter and longer sides.
function _spike(y::T, γ::T) where {T}
    t = y^2 - γ - 1
    x2 = (t + sqrt(max(t^2 - 4γ, zero(T)))) / 2
    cshort = sqrt(max(x2^2 - γ, zero(T)) / (x2^2 + γ * x2))
    clong = sqrt(max(x2^2 - γ, zero(T)) / (x2^2 + x2))
    return sqrt(x2), cshort, clong
end

_shrunk(::HardThreshold, y, x, c1, c2) = y
_shrunk(::FrobeniusShrinker, y, x, c1, c2) = x * c1 * c2
_shrunk(::OperatorShrinker, y, x, c1, c2) = x
_shrunk(::NuclearShrinker, y, x, c1, c2) = x * max(c1 * c2 - sqrt((1 - c1^2) * (1 - c2^2)), zero(x))
_shrunk(s::GramShrinker, y, x, c1, c2) = x * (s.dim == 1 ? c1 : c2)

"""
    η = shrink(bw::Biwhitened, shrinker::Shrinker=FrobeniusShrinker())

Estimates of the singular values of the signal in the scaled matrix, one for each of the
`rank(bw)` singular values above the noise, under the rule `shrinker`. They assume that the
noise of the scaled matrix has variance one (see [`noise_variance`](@ref)).

# Examples

```jldoctest
julia> Y = [40 1 0 2; 38 0 1 1; 1 0 2 1; 0 1 1 3];

julia> bw = biwhiten(Y);

julia> rank(bw)
1

julia> shrink(bw, HardThreshold()) == bw.S[1:1]
true

julia> shrink(bw)[1] < shrink(bw, OperatorShrinker())[1] < bw.S[1]
true
```
"""
function shrink(bw::Biwhitened{T}, shrinker::Shrinker=FrobeniusShrinker()) where {T}
    p, q, rowsshort = _shape(bw)
    γ, sq = T(p) / T(q), sqrt(T(q))
    return map(view(bw.S, 1:rank(bw))) do s
        y = s / sq
        x, cshort, clong = _spike(y, γ)
        c1, c2 = rowsshort ? (cshort, clong) : (clong, cshort)
        sq * _shrunk(shrinker, y, x, c1, c2)
    end
end

"""
    Z = scores(bw::Biwhitened, dim, shrinker::Shrinker=HardThreshold())

Coordinates of the rows (`dim = 1`) or columns (`dim = 2`) of the data on the `rank(bw)`
components above the noise: the singular vectors `bw.U` or `bw.V`, each multiplied by the
corresponding singular value from [`shrink`](@ref)`(bw, shrinker)`. `Z` has one row for each
row (or column) of the data and one column for each component. With the default `shrinker`,
these are the principal component scores of the scaled matrix, in which the noise has
unit variance in each coordinate. Rows and columns whose scale is zero have zero scores.
"""
function scores(bw::Biwhitened, dim::Integer, shrinker::Shrinker=HardThreshold())
    dim in (1, 2) || throw(ArgumentError("`dim` must be 1 or 2, got $dim"))
    return (dim == 1 ? bw.U : bw.V) .* shrink(bw, shrinker)'
end

"""
    X = denoise(bw::Biwhitened, shrinker::Shrinker=FrobeniusShrinker())

An estimate of the expected value of the data, as a dense matrix of the same size as the
data. It is assembled from the components of the scaled matrix above the noise, with
singular values from [`shrink`](@ref)`(bw, shrinker)`, and multiplied by the scales `bw.a`
and `bw.b` to return it to the units of the data.
"""
function denoise(bw::Biwhitened, shrinker::Shrinker=FrobeniusShrinker())
    shrinker isa GramShrinker &&
        throw(ArgumentError("`GramShrinker` estimates inner products and applies to `scores`, not to `denoise`"))
    return (bw.a .* bw.U) * Diagonal(shrink(bw, shrinker)) * (bw.b .* bw.V)'
end
