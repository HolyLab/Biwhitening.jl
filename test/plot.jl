@testset "spectrumplot" begin
    if MAKIE_ABSENT
        @test NOMAKIE_ERROR isa ArgumentError
        @test occursin("load a Makie backend", sprint(showerror, NOMAKIE_ERROR))
    end

    rng = StableRNG(5)
    Y = [rpoisson(rng, x) for x in lowrank(rng, 60, 200, 3)]
    bw = biwhiten(Y)
    fig = spectrumplot(bw)
    @test fig isa Figure
    axes = [c for c in fig.content if c isa Axis]
    @test length(axes) == 2
    # Left: the singular values above and below the edge, and the edge. Right: histogram,
    # law, and edge.
    @test length(axes[1].scene.plots) == 3
    @test length(axes[2].scene.plots) == 3
    @test spectrumplot(bw; figure=(; size=(640, 300))).scene.viewport[].widths == [640, 300]

    # Rows and columns without counts, and a matrix without components above the noise.
    Y0 = copy(Y)
    Y0[2, :] .= 0
    Y0[:, 7] .= 0
    @test spectrumplot(biwhiten(Y0)) isa Figure
    noise = biwhiten([rnormal(rng) for _ in 1:40, _ in 1:90], ones(40, 90))
    @test rank(noise) == 0
    @test length([c for c in spectrumplot(noise).content if c isa Axis][1].scene.plots) == 2
end
