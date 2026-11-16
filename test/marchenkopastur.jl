@testset "Marchenko–Pastur law" begin
    @testset "support and density" begin
        d = MarchenkoPastur(0.25)
        @test extrema(d) == (0.25, 2.25)
        @test minimum(d) == 0.25 && maximum(d) == 2.25
        @test pdf(d, 0.2) == 0 && pdf(d, 2.3) == 0
        @test pdf(d, 1.0) ≈ sqrt(1.25 * 0.75) / (2π * 0.25)
        @test cdf(d, 0.25) == 0 && cdf(d, 0.1) == 0
        @test cdf(d, 2.25) == 1 && cdf(d, 3.0) == 1
    end

    @testset "distribution function is the integral of the density (γ = $γ)" for γ in (0.01, 0.3, 0.9)
        d = MarchenkoPastur(γ)
        lo, hi = extrema(d)
        xs = range(lo, hi; length=100_001)
        # Trapezoid rule; the density vanishes at both ends of the support.
        dens = pdf.(Ref(d), xs)
        integral = cumsum(dens) .* step(xs) .- dens .* step(xs) ./ 2
        @test maximum(abs, integral .- cdf.(Ref(d), xs)) < 1e-5
        @test issorted(cdf.(Ref(d), xs))
    end

    @testset "square matrices" begin
        # At γ = 1 the square roots of the eigenvalues follow the quarter-circle law on [0, 2].
        d = MarchenkoPastur(1.0)
        quartercircle(t) = (t * sqrt(4 - t^2) / 2 + 2 * asin(t / 2)) / π
        for x in (0.01, 0.3, 1.0, 2.5, 3.9)
            @test cdf(d, x) ≈ quartercircle(sqrt(x))
        end
    end

    @testset "quantiles" begin
        for γ in (0.05, 0.5, 1.0)
            d = MarchenkoPastur(γ)
            for p in (0.1, 0.5, 0.9)
                @test cdf(d, quantile(d, p)) ≈ p
            end
            @test median(d) == quantile(d, 0.5)
            @test quantile(d, 0) ≈ minimum(d) atol = 1e-12
            @test quantile(d, 1) ≈ maximum(d)
        end
        # The mean of the law is one, and the median lies below it.
        @test 0.6 < median(MarchenkoPastur(1.0)) < median(MarchenkoPastur(0.1)) < 1
        @test_throws "0 ≤ p ≤ 1" quantile(MarchenkoPastur(0.5), 1.5)
    end

    @testset "construction" begin
        @test MarchenkoPastur(1 // 4) isa MarchenkoPastur{Float64}
        @test MarchenkoPastur(0.25f0) isa MarchenkoPastur{Float32}
        @test cdf(MarchenkoPastur(0.25f0), 1) isa Float32
        @test pdf(MarchenkoPastur(0.25f0), 1) isa Float32
        @test cdf(MarchenkoPastur(0.25f0), 1) ≈ cdf(MarchenkoPastur(0.25), 1) rtol = 1e-5
        @test_throws "0 < γ ≤ 1" MarchenkoPastur(0.0)
        @test_throws "0 < γ ≤ 1" MarchenkoPastur(1.5)
    end

    @testset "eigenvalues of Gaussian noise" begin
        rng = StableRNG(11)
        p, q = 200, 800
        E = [rnormal(rng) for _ in 1:p, _ in 1:q]
        λ = sort(eigvals(Symmetric(E * E')) ./ q)
        d = MarchenkoPastur(p / q)
        @test Biwhitening._ksdistance(λ, d) < 0.02
        @test median(λ) ≈ median(d) rtol = 0.02
    end
end
