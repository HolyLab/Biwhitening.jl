"""
Separation of signal from noise in a data matrix whose noise variance differs from entry to
entry. Biwhitening scales the rows and columns so that the noise has average variance one in
each of them, compares the spectrum of the scaled matrix with the Marchenko–Pastur law,
counts the components above the noise, and shrinks their singular values.

The entry points are [`biwhiten`](@ref) and [`fit_variance`](@ref).
"""
module Biwhitening

using LinearAlgebra: LinearAlgebra, Diagonal, Symmetric, eigen, eigvals, mul!, rank
using MatrixCovers: noise_cover
using Statistics: Statistics, median, quantile

export biwhiten, Biwhitened, scaled, ks_distance, noise_variance, noise_edge, rank
export QuadraticVariance, variance_estimate, fit_variance, VarianceFit
export MarchenkoPastur
export Shrinker, HardThreshold, FrobeniusShrinker, OperatorShrinker, NuclearShrinker, GramShrinker
export shrink, scores, score_noise, denoise

# `public` is parsed as a keyword only from Julia 1.11; this package supports 1.10.
@static if VERSION >= v"1.11"
    eval(Meta.parse("public pdf, cdf"))
end

include("marchenkopastur.jl")
include("variance.jl")
include("biwhiten.jl")
include("fit.jl")
include("shrinkage.jl")

end
