"""
    VarianceFit{T}

The result of [`fit_variance`](@ref). Its fields are

- `model`: the fitted [`QuadraticVariance`](@ref);
- `β`, `α`: the weight of the quadratic term and the overall scale of the chosen variance
  estimate `α * ((1 - β) * y + β * y^2)`;
- `ks`: the Kolmogorov–Smirnov distance between the eigenvalues of the scaled matrix and
  the Marchenko–Pastur law for that estimate;
- `scan`: `β`, `α`, `ks`, and the number of singular values above the noise (`rank`) for
  every candidate `β`, in order of `β`.
"""
struct VarianceFit{T<:AbstractFloat}
    model::QuadraticVariance{T}
    β::T
    α::T
    ks::T
    scan::Vector{@NamedTuple{β::T, α::T, ks::T, rank::Int}}
end

function Base.show(io::IO, fit::VarianceFit{T}) where {T}
    print(io, "VarianceFit{", T, "}(", fit.model, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", fit::VarianceFit{T}) where {T}
    print(io, "VarianceFit{", T, "}: Var(y) = ", round(fit.model.b; sigdigits=4), "*x + ", round(fit.model.c; sigdigits=4), "*x^2")
    print(io, "\n      β       α      KS distance   rank")
    for s in fit.scan
        print(io, "\n  ", s.β == fit.β ? "* " : "  ", rpad(round(s.β; digits=4), 8), rpad(round(s.α; sigdigits=4), 7),
              rpad(round(s.ks; sigdigits=3), 14), s.rank)
    end
    return nothing
end

"""
    fit = fit_variance(Y; grid=0:0.05:1, kwargs...)

Choose a [`QuadraticVariance`](@ref) for the nonnegative data matrix `Y` by the method of
Landa, Zhang, and Kluger (2022, section 5.1), and return a [`VarianceFit`](@ref);
`fit.model` is the chosen model.

For each `β` in `grid`, `Y` is biwhitened with the variance estimate
`(1 - β) * y + β * y^2`, which interpolates between a variance proportional to the expected
value (`β = 0`) and one proportional to its square (`β = 1`). An overall scale `α` is then
set so that the median eigenvalue of the scaled matrix equals the median of the
Marchenko–Pastur law. The fit chooses the `β` whose eigenvalues have the smallest
Kolmogorov–Smirnov distance to the law, and the middle one of several that share that
distance. The chosen estimate, `α * ((1 - β) * y + β * y^2)`,
is the [`variance_estimate`](@ref) of the model with

    a = 0,    b = α * (1 - β) / (1 - α * β),    c = α * β / (1 - α * β)

No such model exists when `α * β ≥ 1`, and the fit then throws an error. `fit.scan` holds the
results for every candidate in order of `β`; a distance that changes little across `grid`
means that the spectrum does not determine `β`.

The keywords are those of [`biwhiten`](@ref). The fit costs one biwhitening for each element
of `grid`.

Because the model is selected for its small distance, `fit.ks` is biased downward. To assess
the fit without this bias, fit the model on some of the rows of `Y`, biwhiten the remaining
rows with it, and compute their [`ks_distance`](@ref).
"""
function fit_variance(Y::AbstractMatrix{<:Real}; grid=0:0.05:1, kwargs...)
    Base.require_one_based_indexing(Y)
    T = float(eltype(Y))
    isempty(grid) && throw(ArgumentError("`grid` must not be empty"))
    all(β -> 0 <= β <= 1, grid) || throw(ArgumentError("every element of `grid` must lie in [0, 1], got $grid"))
    scan = [_scanpoint(Y, β; kwargs...) for β in sort!(unique!(T[β for β in grid]))]
    best = scan[_best(scan)]
    d = 1 - best.α * best.β
    d > 0 || throw(ArgumentError("the best candidate, β = $(best.β) with α = $(best.α), has α * β ≥ 1 and corresponds to no quadratic variance"))
    model = QuadraticVariance{T}(zero(T), best.α * (1 - best.β) / d, best.α * best.β / d)
    return VarianceFit{T}(model, best.β, best.α, best.ks, scan)
end

# Index of the candidate with the smallest distance, and of the middle one when several
# share it. Ties are exact when the largest deviation from the law is the one that the
# components of the signal cause, which is the same for every model that fits the noise.
function _best(scan)
    ksmin = minimum(s -> s.ks, scan)
    ties = findall(s -> s.ks == ksmin, scan)
    return ties[(length(ties)+1)÷2]
end

function _scanpoint(Y, β::T; kwargs...) where {T}
    V = map(Y) do y
        yf = T(y)
        (1 - β) * yf + β * yf^2
    end
    _, _, Yh, _, _ = _normalize(Y, V, T; kwargs...)
    p, q = minmax(size(Yh)...)
    λ = max.(eigvals(_gram(Yh)), zero(T)) ./ q
    mp = MarchenkoPastur{T}(T(p) / T(q))
    α = median(λ) / median(mp)
    λ ./= α
    return (; β, α, ks=_ksdistance(λ, mp), rank=count(>(maximum(mp)), λ))
end
