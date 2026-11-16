"""
    fig = spectrumplot(bw::Biwhitened; figure=(;))

Plot the singular values of the scaled matrix in two panels and return the Makie `Figure`.
This function needs a Makie backend, such as CairoMakie or GLMakie, to be loaded.

The left panel shows every positive singular value against its index, on logarithmic axes,
with [`noise_edge`](@ref) as a dashed line and the `rank(bw)` values above it marked. The
right panel shows a histogram of the singular values up to a little beyond the edge,
together with the density that the Marchenko–Pastur law predicts for noise of unit
variance.

The plot shows what the numbers `rank(bw)`, [`noise_variance`](@ref), and
[`ks_distance`](@ref) summarize. A histogram that follows the curve means that the variance
model describes the noise. In the left panel, singular values that descend into the bulk
without a gap mean that the count of components depends on exactly where the edge lies,
and a change of slope among them may separate components of different origin.

`figure` holds keyword arguments for the `Figure`, such as `(; size=(1000, 450))`.
"""
function spectrumplot(bw::Biwhitened; kwargs...)
    ext = Base.get_extension(@__MODULE__, :BiwhiteningMakieExt)
    ext === nothing &&
        throw(ArgumentError("`spectrumplot` needs Makie; load a Makie backend, such as CairoMakie or GLMakie, first"))
    return ext.spectrumplot(bw; kwargs...)
end
