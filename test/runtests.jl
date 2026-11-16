using Biwhitening
using Biwhitening: cdf, pdf
using LinearAlgebra
using OffsetArrays: OffsetArray
using SparseArrays: SparseMatrixCSC, sparse
using StableRNGs: StableRNG
using Statistics: median, quantile
using Aqua
using ExplicitImports
using Test

include("helpers.jl")

# `spectrumplot` is implemented in an extension that loads with Makie. A session that has
# already loaded Makie cannot observe its absence.
const MAKIE_ABSENT = Base.get_extension(Biwhitening, :BiwhiteningMakieExt) === nothing
const NOMAKIE_ERROR = try
    spectrumplot(biwhiten([3 0 1 2; 0 2 5 1; 4 1 0 3]))
catch err
    err
end
using Makie: Makie, Axis, Figure

@testset "Biwhitening.jl" begin
    include("marchenkopastur.jl")
    include("variance.jl")
    include("biwhiten.jl")
    include("fit.jl")
    include("shrinkage.jl")
    include("plot.jl")

    @testset "Aqua" begin
        Aqua.test_all(Biwhitening)
    end

    @testset "ExplicitImports" begin
        @test check_no_implicit_imports(Biwhitening) === nothing
        @test check_no_stale_explicit_imports(Biwhitening) === nothing
        @test check_all_explicit_imports_via_owners(Biwhitening) === nothing
        @test check_all_qualified_accesses_via_owners(Biwhitening) === nothing
        @test check_no_self_qualified_accesses(Biwhitening) === nothing
        if VERSION >= v"1.11"
            @test check_all_explicit_imports_are_public(Biwhitening) === nothing
            @test check_all_qualified_accesses_are_public(Biwhitening) === nothing
        end
    end
end
