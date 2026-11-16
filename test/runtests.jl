using Biwhitening
using Biwhitening: cdf, pdf
using LinearAlgebra
using OffsetArrays: OffsetArray
using SparseArrays: sparse
using StableRNGs: StableRNG
using Statistics: median, quantile
using Aqua
using ExplicitImports
using Test

include("helpers.jl")

@testset "Biwhitening.jl" begin
    include("marchenkopastur.jl")
    include("variance.jl")
    include("biwhiten.jl")
    include("fit.jl")
    include("shrinkage.jl")

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
