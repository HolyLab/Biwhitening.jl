"""
    MarchenkoPastur(γ)

The Marchenko–Pastur distribution with aspect ratio `0 < γ ≤ 1`: the limiting distribution
of the eigenvalues of `E * E' / q` for a `p × q` matrix `E` of independent entries with mean
zero and variance one, as `p` and `q` grow with `p / q → γ`.

For `d = MarchenkoPastur(γ)`, the support is `extrema(d) == ((1 - √γ)^2, (1 + √γ)^2)`.
[`Biwhitening.pdf`](@ref), [`Biwhitening.cdf`](@ref), `quantile`, and `median` evaluate it.

# Examples

```jldoctest
julia> d = MarchenkoPastur(0.25);

julia> extrema(d)
(0.25, 2.25)

julia> Biwhitening.cdf(d, 2.25)
1.0
```
"""
struct MarchenkoPastur{T<:AbstractFloat}
    γ::T

    function MarchenkoPastur{T}(γ) where {T<:AbstractFloat}
        0 < γ <= 1 || throw(ArgumentError("the aspect ratio must satisfy 0 < γ ≤ 1, got $γ"))
        return new{T}(γ)
    end
end
MarchenkoPastur(γ::Real) = MarchenkoPastur{float(typeof(γ))}(γ)

Base.minimum(d::MarchenkoPastur) = (1 - sqrt(d.γ))^2
Base.maximum(d::MarchenkoPastur) = (1 + sqrt(d.γ))^2
Base.extrema(d::MarchenkoPastur) = (minimum(d), maximum(d))

"""
    Biwhitening.pdf(d::MarchenkoPastur, x)

The density of `d` at `x`.
"""
function pdf(d::MarchenkoPastur{T}, x::Real) where {T}
    lo, hi = extrema(d)
    lo < x < hi || return zero(T)
    return T(sqrt((hi - x) * (x - lo)) / (2 * T(π) * d.γ * x))
end

"""
    Biwhitening.cdf(d::MarchenkoPastur, x)

The probability that a variable with distribution `d` is at most `x`.
"""
function cdf(d::MarchenkoPastur{T}, x::Real) where {T}
    lo, hi = extrema(d)
    x <= lo && return zero(T)
    x >= hi && return one(T)
    γ = d.γ
    s = sqrt((hi - x) * (x - lo))
    # The integral of the density from `lo` to `x`. The last term vanishes at `γ = 1`.
    F = (s - (1 + γ) * atan(1 + γ - x, s) + (1 - γ) * atan((1 - γ)^2 - (1 + γ) * x, (1 - γ) * s)) / (2 * T(π) * γ)
    return clamp(T(F + 1 // 2), zero(T), one(T))
end

"""
    quantile(d::MarchenkoPastur, p)

The point `x` with `Biwhitening.cdf(d, x) == p`, for `0 ≤ p ≤ 1`.
"""
function Statistics.quantile(d::MarchenkoPastur{T}, p::Real) where {T}
    0 <= p <= 1 || throw(ArgumentError("the probability must satisfy 0 ≤ p ≤ 1, got $p"))
    lo, hi = extrema(d)
    # Bisection: the distribution function is continuous and increasing on the support.
    while true
        mid = (lo + hi) / 2
        (lo < mid < hi) || return mid
        if cdf(d, mid) < p
            lo = mid
        else
            hi = mid
        end
    end
end

Statistics.median(d::MarchenkoPastur) = quantile(d, 1 // 2)

# Kolmogorov–Smirnov distance between the empirical distribution of the values `λ`, sorted
# in increasing order, and the distribution `d`.
function _ksdistance(λ::AbstractVector, d::MarchenkoPastur{T}) where {T}
    n = length(λ)
    ks = zero(T)
    for (k, x) in enumerate(λ)
        F = cdf(d, x)
        ks = max(ks, abs(T(k) / n - F), abs(T(k - 1) / n - F))
    end
    return ks
end
