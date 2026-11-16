@testset "variance models" begin
    @testset "construction" begin
        @test QuadraticVariance() === QuadraticVariance(0.0, 1.0, 0.0)
        @test QuadraticVariance(c=1 // 3) === QuadraticVariance(0.0, 1.0, 1 / 3)
        @test QuadraticVariance(1, 2, 3) isa QuadraticVariance{Float64}
        @test QuadraticVariance(1.0f0, 2, 3) isa QuadraticVariance{Float32}
        @test QuadraticVariance{Float32}(QuadraticVariance(c=0.5)) === QuadraticVariance(0.0f0, 1.0f0, 0.5f0)
        @test_throws "no variance estimate exists for c = -1" QuadraticVariance(c=-1)
        @test_throws "must be finite" QuadraticVariance(b=Inf)
    end

    @testset "estimates" begin
        @test variance_estimate(QuadraticVariance(), 3) === 3.0
        @test variance_estimate(QuadraticVariance(a=2, b=0), 7) === 2.0
        @test variance_estimate(QuadraticVariance(a=1, b=2, c=3), 2.0) ≈ (1 + 2 * 2 + 3 * 4) / 4
        # A narrow integer type does not overflow.
        @test variance_estimate(QuadraticVariance(c=1), UInt8(200)) == (200 + 200^2) / 2
        @test variance_estimate(QuadraticVariance{Float32}(0, 1, 0.5), 3) isa Float32
    end

    @testset "estimates are unbiased" begin
        # Negative binomial: a Poisson variable whose mean is `x` times a gamma variable of
        # shape `θ` has variance `x + x^2 / θ`.
        rng = StableRNG(5)
        x, θ = 4.0, 2
        ys = [rpoisson(rng, x * rgamma(rng, θ)) for _ in 1:200_000]
        model = QuadraticVariance(c=1 / θ)
        estimate = sum(y -> variance_estimate(model, y), ys) / length(ys)
        @test estimate ≈ x + x^2 / θ rtol = 0.03
        @test sum(abs2, ys .- x) / length(ys) ≈ x + x^2 / θ rtol = 0.03
    end
end
