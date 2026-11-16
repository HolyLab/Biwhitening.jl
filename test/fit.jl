@testset "fit_variance" begin
    rng = StableRNG(2)
    m, n, r = 300, 1000, 10
    X = lowrank(rng, m, n, r)

    @testset "negative binomial counts" begin
        # Dispersion 3: Var(y) = x + x^2 / 3, which the estimate with α = 1, β = 1/4 matches.
        Y = [rpoisson(rng, x * rgamma(rng, 3)) for x in X]
        fit = fit_variance(Y)
        @test fit isa VarianceFit{Float64}
        @test length(fit.scan) == 21
        @test [s.β for s in fit.scan] == 0:0.05:1
        @test 0.2 <= fit.β <= 0.3
        @test fit.α ≈ 1 atol = 0.15
        @test fit.model.a == 0
        @test fit.model.b ≈ 1 atol = 0.2
        @test fit.model.c ≈ 1 / 3 atol = 0.1
        @test fit.ks == minimum(s.ks for s in fit.scan)
        best = only(s for s in fit.scan if s.β == fit.β)
        @test (best.α, best.ks) == (fit.α, fit.ks)
        # The fitted model reproduces the chosen candidate.
        bw = biwhiten(Y, fit.model)
        @test noise_variance(bw) ≈ 1 atol = 1e-8
        @test ks_distance(bw) ≈ fit.ks
        @test rank(bw) == best.rank
        @test variance_estimate(fit.model, 3) ≈ fit.α * ((1 - fit.β) * 3 + fit.β * 9)
        # The distance has a single minimum along the grid.
        ks = [s.ks for s in fit.scan]
        k = argmin(ks)
        @test issorted(ks[1:k]; rev=true) && issorted(ks[k:end])
    end

    @testset "Poisson counts" begin
        Y = [rpoisson(rng, x) for x in X]
        fit = fit_variance(Y; grid=[0, 0.1, 0.5, 1])
        @test fit.β == 0
        @test fit.α ≈ 1 atol = 0.05
        @test fit.model.b == fit.α && fit.model.c == 0
        @test [s.β for s in fit.scan] == [0, 0.1, 0.5, 1]
        @test first(fit.scan).rank == r
        @test fit_variance(sparse(Y); grid=[0.0]).α ≈ fit.α
        @test fit_variance(Float32.(Y); grid=[0.0]) isa VarianceFit{Float32}
        # A scale that the model leaves free is recovered.
        fit5 = fit_variance(5 .* Y; grid=[0.0])
        @test fit5.model.b ≈ 5 * fit.α
        @test rank(biwhiten(5 .* Y, fit5.model)) == r
    end

    @testset "arguments" begin
        Y = [rpoisson(rng, x) for x in X]
        @test_throws "`grid` must not be empty" fit_variance(Y; grid=Float64[])
        @test_throws "must lie in [0, 1]" fit_variance(Y; grid=[0.5, 1.5])
        @test_throws "offset arrays are not supported" fit_variance(OffsetArray(Y, -1, -1))
    end

    @testset "display" begin
        Y = [rpoisson(rng, x) for x in X]
        fit = fit_variance(Y; grid=[0, 0.5])
        str = sprint(show, MIME"text/plain"(), fit)
        @test startswith(str, "VarianceFit{Float64}: Var(y) = ")
        @test count(==('\n'), str) == 3
        @test occursin("\n  * 0.0 ", str) && occursin("\n    0.5 ", str)
        @test sprint(show, fit) == "VarianceFit{Float64}($(fit.model))"
    end
end
