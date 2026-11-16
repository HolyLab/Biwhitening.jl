@testset "shrinkage" begin
    # A signal of rank three in Gaussian noise of unit variance. `xs .* √q` are its singular
    # values; a component is detectable when `x > γ^(1/4)`.
    rng = StableRNG(3)
    p, q = 400, 1000
    γ = p / q
    xs = [3.0, 2.0, 1.2]
    Us, Vs = orthonormal(rng, p, 3), orthonormal(rng, q, 3)
    X = Us * Diagonal(xs .* sqrt(q)) * Vs'
    Y = X .+ [rnormal(rng) for _ in 1:p, _ in 1:q]
    bw = biwhiten(Y, ones(p, q))
    @test rank(bw) == 3
    y = bw.S[1:3] ./ sqrt(q)

    shrinkers = (HardThreshold(), FrobeniusShrinker(), OperatorShrinker(), NuclearShrinker())

    @testset "signal strength and angles" begin
        x = shrink(bw, OperatorShrinker()) ./ sqrt(q)
        @test x ≈ xs rtol = 0.03
        # The observed singular value of a component of strength `x`.
        @test y ≈ @. sqrt((1 + x^2) * (γ + x^2)) / x
        c1 = shrink(bw, GramShrinker(1)) ./ shrink(bw, OperatorShrinker())
        c2 = shrink(bw, GramShrinker(2)) ./ shrink(bw, OperatorShrinker())
        @test c1 ≈ abs.(diag(Us' * bw.U)) atol = 0.02
        @test c2 ≈ abs.(diag(Vs' * bw.V)) atol = 0.02
        # The singular vectors of the shorter side are estimated better.
        @test all(c1 .> c2)
    end

    @testset "closed forms" begin
        x = shrink(bw, OperatorShrinker()) ./ sqrt(q)
        @test shrink(bw, HardThreshold()) == bw.S[1:3]
        @test shrink(bw) == shrink(bw, FrobeniusShrinker())
        @test shrink(bw, FrobeniusShrinker()) ./ sqrt(q) ≈ @. sqrt((y^2 - γ - 1)^2 - 4γ) / y
        @test shrink(bw, NuclearShrinker()) ./ sqrt(q) ≈ @. max(x^4 - γ - sqrt(γ) * x * y, 0) / (x^2 * y)
        for s in (shrinkers..., GramShrinker(1), GramShrinker(2))
            η = shrink(bw, s)
            @test length(η) == 3 && issorted(η; rev=true)
            @test all(0 .<= η .<= bw.S[1:3])
        end
        @test all(shrink(bw, NuclearShrinker()) .<= shrink(bw, FrobeniusShrinker()) .<= shrink(bw, OperatorShrinker()))
    end

    @testset "at the edge of the noise" begin
        # A singular value at the edge carries no information about the signal: the
        # estimated vectors are orthogonal to the true ones.
        edge = 1 + sqrt(γ)
        x, cshort, clong = Biwhitening._spike(edge, γ)
        # The cosines are square roots of differences that vanish at the edge.
        @test x ≈ γ^(1 / 4) atol = 1e-6
        @test cshort ≈ 0 atol = 1e-3
        @test clong ≈ 0 atol = 1e-3
        for s in (FrobeniusShrinker(), NuclearShrinker(), GramShrinker(1))
            @test Biwhitening._shrunk(s, edge, x, cshort, clong) ≈ 0 atol = 1e-3
        end
        # A strong component is nearly unchanged.
        x, cshort, clong = Biwhitening._spike(1000.0, γ)
        @test x ≈ 1000 rtol = 1e-5
        @test cshort ≈ 1 atol = 1e-5
        @test clong ≈ 1 atol = 1e-5
    end

    @testset "each rule is the best for its loss" begin
        errors(loss) = [loss(denoise(bw, s) .- X) for s in shrinkers]
        @test argmin(errors(norm)) == 2
        @test argmin(errors(opnorm)) == 3
        @test argmin(errors(D -> sum(svdvals(D)))) == 4
        # Inner products between the rows, and between the columns, of the signal.
        rules = (shrinkers..., GramShrinker(1), GramShrinker(2))
        rowerr = [norm(scores(bw, 1, s) * scores(bw, 1, s)' .- X * X') for s in rules]
        @test argmin(rowerr) == 5
        colerr = [norm(scores(bw, 2, s) * scores(bw, 2, s)' .- X' * X) for s in rules]
        @test argmin(colerr) == 6
    end

    @testset "scores and denoising" begin
        @test scores(bw, 1) == bw.U .* bw.S[1:3]'
        @test scores(bw, 2) == bw.V .* bw.S[1:3]'
        @test scores(bw, 1, FrobeniusShrinker()) == bw.U .* shrink(bw)'
        @test size(scores(bw, 1)) == (p, 3) && size(scores(bw, 2)) == (q, 3)
        @test denoise(bw) == denoise(bw, FrobeniusShrinker())
        @test denoise(bw, HardThreshold()) ≈ bw.U * Diagonal(bw.S[1:3]) * bw.V'
        @test_throws "`dim` must be 1 or 2" scores(bw, 3)
        @test_throws "`dim` must be 1 or 2" GramShrinker(0)
        @test_throws "applies to `scores`, not to `denoise`" denoise(bw, GramShrinker(1))
    end

    @testset "noise in the scores" begin
        c1 = shrink(bw, GramShrinker(1)) ./ shrink(bw, OperatorShrinker())
        c2 = shrink(bw, GramShrinker(2)) ./ shrink(bw, OperatorShrinker())
        @test score_noise(bw, 1) ≈ bw.S[1:3] .^ 2 .* (1 .- c1 .^ 2) ./ p
        @test score_noise(bw, 2) ≈ bw.S[1:3] .^ 2 .* (1 .- c2 .^ 2) ./ q
        @test score_noise(bw, 1) == score_noise(bw, 1, HardThreshold())
        # The part of each column of the scores that lies outside the signal.
        for (dim, W) in ((1, Us), (2, Vs))
            Z = scores(bw, dim)
            outside = Z .- W * (W' * Z)
            @test vec(sum(abs2, outside; dims=1)) ./ size(Z, 1) ≈ score_noise(bw, dim) rtol = 0.1
        end
        # Weaker components carry more noise, and a strong one carries the noise of the data.
        @test issorted(score_noise(bw, 1))
        @test all(>(1), score_noise(bw, 1))
        strong = biwhiten(Y .+ 30 .* X, ones(p, q))
        @test score_noise(strong, 1) ≈ ones(3) rtol = 0.01
        @test score_noise(strong, 2) ≈ ones(3) rtol = 0.01
        # Shrinking a column of the scores shrinks its noise with it.
        for s in (FrobeniusShrinker(), OperatorShrinker(), GramShrinker(1))
            w = shrink(bw, s) ./ bw.S[1:3]
            @test score_noise(bw, 1, s) ≈ w .^ 2 .* score_noise(bw, 1)
        end
        @test_throws "`dim` must be 1 or 2" score_noise(bw, 0)
    end

    @testset "taller than wide" begin
        bwt = biwhiten(permutedims(Y), ones(q, p))
        for s in shrinkers
            @test shrink(bwt, s) ≈ shrink(bw, s)
            @test denoise(bwt, s) ≈ denoise(bw, s)'
        end
        @test shrink(bwt, GramShrinker(1)) ≈ shrink(bw, GramShrinker(2))
        @test shrink(bwt, GramShrinker(2)) ≈ shrink(bw, GramShrinker(1))
        @test score_noise(bwt, 1) ≈ score_noise(bw, 2)
        @test score_noise(bwt, 2) ≈ score_noise(bw, 1)
    end

    @testset "counts return to the units of the data" begin
        rng = StableRNG(4)
        Xc = lowrank(rng, 300, 1000, 10)
        Yc = [rpoisson(rng, x) for x in Xc]
        bc = biwhiten(Yc)
        Xhat = denoise(bc)
        @test size(Xhat) == size(Yc)
        @test norm(Xhat .- Xc) < 0.3 * norm(Yc .- Xc)
        @test norm(Xhat .- Xc) < norm(denoise(bc, HardThreshold()) .- Xc)
        @test sum(Xhat) ≈ sum(Yc) rtol = 0.01
        @test biwhiten(Float32.(Yc)) |> denoise isa Matrix{Float32}
    end
end
