@testset "biwhiten" begin
    rng = StableRNG(1)
    m, n, r = 300, 1000, 10
    X = lowrank(rng, m, n, r)
    Y = [rpoisson(rng, x) for x in X]
    bw = biwhiten(Y)

    @testset "Poisson counts of known rank" begin
        @test bw isa Biwhitened{Float64}
        @test size(bw) == (m, n) && size(bw, 1) == m && size(bw, 2) == n
        @test rank(bw) == r
        @test noise_variance(bw) ≈ 1 atol = 0.05
        # The components of the signal are `r` of the `m` eigenvalues, and all lie above the
        # support of the law.
        @test r / m - 1e-12 <= ks_distance(bw) < r / m + 0.02
        @test noise_edge(bw) == sqrt(m) + sqrt(n)
        @test MarchenkoPastur(bw) === MarchenkoPastur(m / n)
        @test length(bw.S) == m && issorted(bw.S; rev=true)
        @test count(>(noise_edge(bw)), bw.S) == r
    end

    @testset "scaling equations" begin
        Vhat = Y ./ (bw.a .* bw.b') .^ 2
        @test all(≈(n), sum(Vhat; dims=2))
        @test all(≈(m), sum(Vhat; dims=1))
    end

    @testset "scaled matrix" begin
        @test scaled(bw, Y) ≈ Y ./ (bw.a .* bw.b')
        @test scaled(bw, sparse(Y)) isa SparseMatrixCSC
        @test scaled(bw, sparse(Y)) ≈ scaled(bw, Y)
        @test_throws DimensionMismatch scaled(bw, Y[:, 1:end-1])
        @test_throws "must have the size of the biwhitened matrix" scaled(bw, Y[:, 1:end-1])
        @test_throws "offset arrays are not supported" scaled(bw, OffsetArray(Y, -1, -1))
    end

    @testset "singular values and vectors" begin
        Yhat = Y ./ (bw.a .* bw.b')
        F = svd(Yhat)
        @test bw.S ≈ F.S
        @test size(bw.U) == (m, r) && size(bw.V) == (n, r)
        @test bw.U' * bw.U ≈ I
        @test bw.V' * bw.V ≈ I
        @test bw.U * Diagonal(bw.S[1:r]) * bw.V' ≈ F.U[:, 1:r] * Diagonal(F.S[1:r]) * F.Vt[1:r, :]
    end

    @testset "transposed data" begin
        bwt = biwhiten(permutedims(Y))
        @test bwt.S ≈ bw.S
        @test rank(bwt) == r
        @test bwt.a .* bwt.b' ≈ (bw.a .* bw.b')'
        @test bwt.V * Diagonal(bwt.S[1:r]) * bwt.U' ≈ bw.U * Diagonal(bw.S[1:r]) * bw.V'
        @test noise_variance(bwt) ≈ noise_variance(bw)
        @test ks_distance(bwt) ≈ ks_distance(bw)
    end

    @testset "storage and element types" begin
        for Z in (sparse(Y), view(Y, :, :), permutedims(Y)', sparse(permutedims(Y))', UInt8.(Y), Float64.(Y))
            bz = biwhiten(Z)
            @test bz isa Biwhitened{Float64}
            @test bz.S ≈ bw.S
            @test bz.a .* bz.b' ≈ bw.a .* bw.b'
            @test bz.U * bz.U' ≈ bw.U * bw.U'
            @test bz.V * bz.V' ≈ bw.V * bw.V'
        end
        bw32 = biwhiten(Float32.(Y))
        @test bw32 isa Biwhitened{Float32}
        @test bw32.S[1:r] ≈ bw.S[1:r] rtol = 1e-4
        @test rank(bw32) == r
        @test noise_variance(bw32) isa Float32
        # The quadratic term of a narrow integer type does not overflow.
        model = QuadraticVariance(c=0.5)
        @test maximum(Y) > 16
        @test biwhiten(UInt8.(Y), model).S ≈ biwhiten(Float64.(Y), model).S
        @test_throws "offset arrays are not supported" biwhiten(OffsetArray(Y, -1, -1))
    end

    @testset "empty rows and columns" begin
        rows, cols = [1:119; 121:m+1], [1:6; 8:499; 501:n+2]
        Z = zeros(Int, m + 1, n + 2)
        Z[rows, cols] = Y
        for bz in (biwhiten(Z), biwhiten(sparse(Z)))
            @test size(bz) == (m + 1, n + 2)
            @test bz.S ≈ bw.S
            @test rank(bz) == r
            @test iszero(bz.a[120]) && iszero(bz.b[7]) && iszero(bz.b[500])
            @test bz.a[rows] .* bz.b[cols]' ≈ bw.a .* bw.b'
            @test iszero(bz.U[120, :]) && iszero(bz.V[[7, 500], :])
            @test bz.U[rows, :] * bz.U[rows, :]' ≈ bw.U * bw.U'
            @test noise_edge(bz) == noise_edge(bw)
            @test noise_variance(bz) ≈ noise_variance(bw)
            @test ks_distance(bz) ≈ ks_distance(bw)
            @test iszero(scores(bz, 1)[120, :])
            @test denoise(bz)[rows, cols] ≈ denoise(bw)
            @test iszero(denoise(bz)[120, :]) && iszero(denoise(bz)[:, 7])
            Zh = scaled(bz, Z)
            @test Zh[rows, cols] ≈ scaled(bw, Y)
            @test iszero(Zh[120, :]) && iszero(Zh[:, [7, 500]])
            @test occursin("300×1000 without empty rows and columns", sprint(show, MIME"text/plain"(), bz))
        end
        @test_throws "every variance estimate is zero" biwhiten(zeros(3, 4))
    end

    @testset "variance estimates as a matrix" begin
        @test biwhiten(Y, Y).S ≈ bw.S
        # Constant variance: the scales are constant and the matrix is divided by the
        # standard deviation.
        E = [3 * rnormal(rng) for _ in 1:50, _ in 1:80]
        be = biwhiten(E, fill(9, size(E)))
        @test all(≈(3), be.a .* be.b')
        @test be.S ≈ svdvals(E) ./ 3
        @test rank(be) == 0 && size(be.U) == (50, 0) && size(be.V) == (80, 0)
        @test noise_variance(be) ≈ 1 atol = 0.15
        @test isempty(shrink(be)) && size(scores(be, 2)) == (80, 0)
        @test iszero(denoise(be))
        # Rescaling the rows and columns of the data, with the variances to match, changes
        # the scales and nothing else.
        dr, dc = 0.5 .+ rand(rng, m), 0.5 .+ rand(rng, n)
        bs = biwhiten(dr .* Y .* dc', dr .^ 2 .* Y .* (dc .^ 2)')
        @test bs.S ≈ bw.S
        @test bs.a .* bs.b' ≈ dr .* (bw.a .* bw.b') .* dc'
        @test_throws DimensionMismatch biwhiten(Y, Y[:, 1:end-1])
        @test_throws "must have the size of the data" biwhiten(Y, Y[:, 1:end-1])
    end

    @testset "keywords reach noise_cover" begin
        # A row with a single nonzero entry admits no scaling unless the variances are shrunk.
        Z = copy(Y)
        Z[1, :] .= 0
        Z[1, 1] = 5
        @test_throws "shrink" biwhiten(Z)
        bz = biwhiten(Z; shrink=0.1)
        @test all(>(0), bz.a)
        @test abs(rank(bz) - r) <= 1
    end

    @testset "overdispersed counts under the Poisson model" begin
        Ynb = [rpoisson(rng, x * rgamma(rng, 3)) for x in X]
        bnb = biwhiten(Ynb)
        @test noise_variance(bnb) > 2
        @test ks_distance(bnb) > 0.3
        @test rank(bnb) > 5r
        good = biwhiten(Ynb, QuadraticVariance(c=1 / 3))
        @test noise_variance(good) ≈ 1 atol = 0.05
        @test ks_distance(good) < 0.05
        @test rank(good) <= r
    end

    @testset "display" begin
        str = sprint(show, MIME"text/plain"(), bw)
        @test startswith(str, "Biwhitened{Float64} of a 300×1000 matrix\n  rank: 10\n")
        @test occursin("noise variance: 1.0", str)
        @test occursin("Kolmogorov–Smirnov distance to the Marchenko–Pastur law: 0.03", str)
        @test !occursin("if the noise has the estimated variance", str)
        # Counts that are more variable than the model allows: the display also gives the
        # rank for noise of the estimated variance.
        b2 = biwhiten(2 .* Y)
        @test noise_variance(b2) ≈ 2 atol = 0.1
        @test rank(b2) > 10r
        @test occursin("\n  rank: $(rank(b2)) ($r if the noise has the estimated variance)\n", sprint(show, MIME"text/plain"(), b2))
        @test sprint(show, bw) == "Biwhitened{Float64}(300×1000, rank 10)"
        @test sprint(show, [bw]) == "Biwhitened{Float64}[Biwhitened{Float64}(300×1000, rank 10)]"
    end
end
