"""
    QuadraticVariance(a, b, c)
    QuadraticVariance(; a=0, b=1, c=0)

A noise model in which the variance of each entry `y` of the data is a quadratic function of
its expected value `x`:

    Var(y) = a + b*x + c*x^2

The default, `QuadraticVariance()`, is the Poisson distribution. Other members are the
negative binomial with dispersion `θ` (`c = 1/θ`), the binomial with `n` trials
(`c = -1/n`), the gamma distribution with shape `k` (`b = 0`, `c = 1/k`), and noise of
constant variance `a` (`b = c = 0`). `c = -1` is excluded, because the variance cannot then
be estimated from a single observation.

[`variance_estimate`](@ref) estimates the variance of an entry from its value, and
[`fit_variance`](@ref) chooses `b` and `c` from the data.
"""
struct QuadraticVariance{T<:AbstractFloat}
    a::T
    b::T
    c::T

    function QuadraticVariance{T}(a, b, c) where {T<:AbstractFloat}
        all(isfinite, (a, b, c)) || throw(ArgumentError("the coefficients must be finite, got a = $a, b = $b, c = $c"))
        c == -1 && throw(ArgumentError("no variance estimate exists for c = -1"))
        return new{T}(a, b, c)
    end
end
QuadraticVariance(a::Real, b::Real, c::Real) = QuadraticVariance{float(promote_type(typeof(a), typeof(b), typeof(c)))}(a, b, c)
QuadraticVariance(; a::Real=0.0, b::Real=1.0, c::Real=0.0) = QuadraticVariance(a, b, c)
QuadraticVariance{T}(model::QuadraticVariance) where {T<:AbstractFloat} = QuadraticVariance{T}(model.a, model.b, model.c)

"""
    variance_estimate(model::QuadraticVariance, y)

An unbiased estimate of the variance of an observation whose value is `y`:

    (a + b*y + c*y^2) / (1 + c)

Its expected value is `a + b*x + c*x^2`, where `x` is the expected value of `y`.

# Examples

```jldoctest
julia> variance_estimate(QuadraticVariance(), 3)
3.0

julia> variance_estimate(QuadraticVariance(c=0.5), 3)
5.0
```
"""
function variance_estimate(model::QuadraticVariance{T}, y::Real) where {T}
    # Convert before squaring: `y^2` overflows narrow integer types such as `UInt8`.
    yf = convert(float(promote_type(T, typeof(y))), y)
    return (model.a + model.b * yf + model.c * yf^2) / (1 + model.c)
end

# The matrix of variance estimates for the data `Y`, in the floating-point type `T`.
function _variances(model::QuadraticVariance, Y::AbstractMatrix{<:Real}, ::Type{T}) where {T}
    mT = QuadraticVariance{T}(model)
    return map(y -> variance_estimate(mT, T(y)), Y)
end
