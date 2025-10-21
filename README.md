# SteinVariationalGradientDescent.jl

`SteinVariationalGradientDescent.jl` implements Stein Variational Gradient Descent for differentiable
target densities. The package focuses on type-stable kernels, analytic updates for Gaussian kernels,
and smooth integration with [`SimpleKernelRegression.jl`](https://github.com/NilsWildt/SimpleKernelRegression.jl)
workflows. 

## Highlights

- Type-stable particle updates with in-place and out-of-place drivers
- Median heuristic bandwidth selection paired with progress logging
- Uses `SimpleKernelRegression.jl` kernels 

## Installation

Add the package to an active environment either in development mode

```julia
] dev https://github.com/<user>/SteinVariationalGradientDescent.jl
```

## Quick start

```julia
using SteinVariationalGradientDescent
using SimpleKernelRegression
using Random

rng = Random.MersenneTwister(2024)
μ = [0.5, -1.0]
Σ = [1.0 0.3; 0.3 0.5]

∇log_p(x) = -(Σ \ (x .- μ))
inits = randn(rng, 128, 2)

result = svgd(∇log_p, inits;
              n_iter = 450,
              stepsize = 0.25,
              bandwidth_method = :median,
              kernel = SimpleKernelRegression.Gaussian(1.0),
              progress = true)

result.particles      # final particle cloud (128 × 2)
result.distances      # diagnostic mean pairwise distances per iteration
```

`SVGD.svgd!` updates a particle matrix in-place so that output storage can be reused in long chains or
inside nested optimisation loops.

## Monitoring convergence

`result.distances` collects the mean pairwise distance between particles at every iteration. In practice a
monotonically decreasing trace followed by a plateau indicates that the particle cloud has stabilised.
Pair it with additional problem-specific diagnostics (log posterior, moments) for production workloads.

## Testing

The package ships with a `TestItems.jl` suite that exercises convergence properties, floating-point
stability, and the median bandwidth heuristic. Run it in a Julia session via

```julia
] test SteinVariationalGradientDescent
```

## Documentation

The `docs/` folder contains a `Documenter.jl` setup. Build the documentation locally with

```julia
julia --project=docs docs/make.jl
```

which renders the manual to `docs/build`.

## Example notebooks & scripts

An executable example lives in `examples/gaussian_demo.jl`, recreating the Gaussian test case with a
UnicodePlots scatter plot for terminal use.

## Limitations 

Be aware about recent publications showing variance collapse in high dimensions, see 

* Ba, Jimmy, et al. "Understanding the variance collapse of SVGD in high dimensions." *International Conference on Learning Representations*. 2021.

## References

- Q. Liu and D. Wang (2016), *Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm*, NeurIPS.
- R. Shu, X. Meng, and S. Ermon (2018), *Amortized Inference Regularization*, NeurIPS.
