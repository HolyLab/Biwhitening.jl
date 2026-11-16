```@meta
CurrentModule = Biwhitening
```

# Reference

## Biwhitening

```@docs
biwhiten
Biwhitened
scaled
noise_edge
noise_variance
ks_distance
```

## Variance models

```@docs
QuadraticVariance
variance_estimate
fit_variance
VarianceFit
```

## The Marchenko–Pastur law

```@docs
MarchenkoPastur
MarchenkoPastur(::Biwhitened)
Biwhitening.pdf
Biwhitening.cdf
Statistics.quantile(::MarchenkoPastur, ::Real)
```

## Shrinkage

```@docs
Shrinker
HardThreshold
FrobeniusShrinker
OperatorShrinker
NuclearShrinker
GramShrinker
shrink
scores
score_noise
denoise
```

## Module

```@docs
Biwhitening.Biwhitening
```
