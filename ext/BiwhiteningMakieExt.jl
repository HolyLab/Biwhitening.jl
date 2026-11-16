module BiwhiteningMakieExt

using Biwhitening: Biwhitening, Biwhitened, MarchenkoPastur, noise_edge
using LinearAlgebra: rank
using Makie: Axis, Figure, axislegend, hist!, hlines!, lines!, scatter!, vlines!, xlims!

# Right limit of the histogram, in units of the noise edge, and its number of bins.
const BULKLIMIT = 1.35
const NBINS = 90

function spectrumplot(bw::Biwhitened; figure=(;))
    S = bw.S
    k = rank(bw)
    edge = noise_edge(bw)
    q = max(count(!iszero, bw.a), count(!iszero, bw.b))
    mp = MarchenkoPastur(bw)

    fig = Figure(; size=(1200, 500), figure...)

    ax1 = Axis(fig[1, 1]; xlabel="index", ylabel="singular value", xscale=log10, yscale=log10,
               title="Singular values of the scaled matrix")
    # The logarithmic axis cannot show singular values that are zero; they come last.
    npos = something(findlast(>(0), S), 0)
    k > 0 && scatter!(ax1, 1:k, S[1:k]; color=:firebrick, markersize=8, label="above the noise edge ($k)")
    npos > k && scatter!(ax1, k+1:npos, S[k+1:npos]; color=:gray40, markersize=4, label="below")
    hlines!(ax1, [edge]; color=:black, linestyle=:dash, label="noise edge, $(round(edge; sigdigits=4))")
    axislegend(ax1; position=:rt)

    hi = BULKLIMIT * edge
    nbeyond = count(>(hi), S)
    title = "Bulk and the Marchenko–Pastur law"
    nbeyond > 0 && (title *= " ($nbeyond values beyond the right limit)")
    ax2 = Axis(fig[1, 2]; xlabel="singular value", ylabel="density", title)
    inside = filter(<=(hi), S)
    # Each value weighs `1 / (length(S) * binwidth)`, so the histogram is a density over all
    # the singular values and not only over those in view.
    hist!(ax2, inside; bins=range(0, hi; length=NBINS + 1), weights=fill(NBINS / (length(S) * hi), length(inside)),
          color=(:steelblue, 0.55), label="singular values")
    # A singular value `s` corresponds to the eigenvalue `s^2 / q`, whose density is the law's.
    ss = range(sqrt(q * minimum(mp)), sqrt(q * maximum(mp)); length=600)
    lines!(ax2, ss, [Biwhitening.pdf(mp, s^2 / q) * 2s / q for s in ss]; color=:black, linewidth=2,
           label="Marchenko–Pastur law")
    vlines!(ax2, [edge]; color=:black, linestyle=:dash, label="noise edge")
    xlims!(ax2, 0, hi)
    axislegend(ax2; position=:lt)

    return fig
end

end
