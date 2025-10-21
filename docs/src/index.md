# SteinVariationalGradientDescent.jl

`SteinVariationalGradientDescent.jl` provides Stein Variational Gradient Descent with explicit control over kernel bandwidth,
progress logging, and type-stable updates. The implementation leans on the `Kernelregression` module
from [`SimpleKernelRegression.jl`](https://github.com/NilsWildt/SimpleKernelRegression.jl) for kernels and distances
and takes advantage of analytic gradients for Gaussian kernels. Other kernels fall back to automatic differentiation via Enzyme.

## Highlights

- Type-stable particle updates with a pure functional driver.
- Median heuristic bandwidth selection and progress logging for long runs.
- Drop-in compatibility with `Kernelregression` kernels and diagnostics.

The package exports a short alias `SVGD` for convenience alongside the following entry points:

- [`SVGD.svgd`](@ref) – functional driver returning a named tuple with `particles` and `distances`.
- [`SVGD.median_heuristic`](@ref) – helper used for Gaussian bandwidth selection.
- [`SVGD.compute_mean_distance`](@ref) – diagnostic mean distance.

## Quick example

```julia
using LinearAlgebra: norm
using Random
using SimpleKernelRegression
using SteinVariationalGradientDescent

rng = Random.MersenneTwister(1)
μ = [0.0, 1.0]
Σ = [1.0 0.3; 0.3 2.0]

∇log_p(x) = -(Σ \ (x .- μ))
inits = randn(rng, 256, 2)

kernel = SimpleKernelRegression.Gaussian(1.0)
result = svgd(∇log_p, inits;
              n_iter = 400,
              stepsize = 0.2,
              bandwidth_method = :median,
              kernel = kernel,
              progress = true)

mean_particles = vec(mean(result.particles, dims = 1))
println("Mean error = ", norm(mean_particles - μ))
```

`result.distances` stores the mean pairwise distance across iterations and is a lightweight way to
monitor particle collapse or divergence. Plateauing values typically indicate the sampler has converged.

## Kernel bandwidth options

The default `bandwidth_method = :median` recomputes a Gaussian kernel with the median heuristic

``σ = \sqrt{\operatorname{median}(\|x_i - x_j\|^2) / \log(n + 1)}``

at the start of every run. Switch to `:manual` to keep the kernel as provided by the caller.

## API reference

```@docs
svgd
median_heuristic
compute_mean_distance
```
