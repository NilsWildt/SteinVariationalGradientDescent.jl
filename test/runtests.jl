using SteinVariationalGradientDescent
using TestItems
using TestItemRunner

const PKGROOT = normpath(joinpath(@__DIR__, ".."))

@testitem "Gaussian posterior recovery" tags=[:stochastic] begin
    using LinearAlgebra
    using Random
    using SimpleKernelRegression
    using Statistics

    rng = Random.MersenneTwister(42)
    μ = [0.5, -1.0]
    Σ = [1.0 0.3; 0.3 0.5]

    function ∇log_p(x)
        return -(Σ \ (x .- μ))
    end

    inits = randn(rng, 96, 2) .* 1.5 .+ 2
    kernel = SimpleKernelRegression.Gaussian(1.0)
    result = svgd(∇log_p, inits; n_iter=1000, stepsize=0.01, bandwidth_method=:median, kernel=kernel, progress=true)

    final = result.particles
    μ̂ = vec(mean(final, dims=1))
    Σ̂ = cov(final)

    mean_err = norm(μ̂ - μ)
    cov_err = norm(Σ̂ - Σ, 2)

    @test mean_err < 0.45
    @test cov_err < 1.0
    dists = result.distances
    tail_idx = max(length(dists) - 15, 1)
    @test dists[end] < dists[1]
    @test abs(dists[end] - dists[tail_idx]) < 0.15
end

@testitem "Stepsize impacts convergence" begin
    using LinearAlgebra
    using Random
    using SimpleKernelRegression
    using Statistics
    rng = Random.MersenneTwister(31)
    ∇log_p(x) = -x
    inits = randn(rng, 96, 1) .* 4

    kernel = SimpleKernelRegression.Gaussian(1.0)
    small = svgd(∇log_p, inits; n_iter=250, stepsize=0.05, bandwidth_method=:median, kernel=kernel, progress=false)
    large = svgd(∇log_p, inits; n_iter=250, stepsize=0.25, bandwidth_method=:median, kernel=kernel, progress=false)

    @test abs(mean(large.particles)) < abs(mean(small.particles))
    @test large.distances[end] < small.distances[end]
end
@testitem "svgd preserves inputs" begin
    using Random
    using SimpleKernelRegression
    rng = Random.MersenneTwister(99)
    ∇log_p(x) = -x
    inits = randn(rng, Float32, 48, 3)

    inits_copy = copy(inits)
    kernel = SimpleKernelRegression.Gaussian(1.0)
    result = svgd(∇log_p, inits; n_iter=200, stepsize=0.15, bandwidth_method=:median, kernel=kernel, progress=false)

    @test inits == inits_copy
    @test result.particles !== inits
    @test eltype(result.particles) === Float32
    @test eltype(result.distances) === Float64
end

@testitem "Median heuristic edge cases" begin
    single = ones(Float64, 1, 3)
    @test median_heuristic(single) == 0.0
    @test compute_mean_distance(single) == 0.0

    pair = [0.0 0.0; 3.0 4.0]
    @test median_heuristic(pair) > 0
    @test compute_mean_distance(pair) ≈ 5.0 atol=1e-8
end

@testitem "Type inference" begin
    using Random
    using SimpleKernelRegression

    rng = Random.MersenneTwister(7)
    ∇log_p(x) = -x
    inits = randn(rng, 32, 2)
    kernel = SimpleKernelRegression.Gaussian(1.0)
    result = @inferred svgd(∇log_p, inits; n_iter=25, stepsize=0.15,
                             bandwidth_method=:median, kernel=kernel, progress=false)
    @test result isa NamedTuple{(:particles, :distances)}
    @test result.particles isa Matrix{Float64}
    @test result.distances isa Vector{Float64}
end

const ALLOWED_PREFIXES = map(dir -> normpath(joinpath(PKGROOT, dir)), ("src", "test"))

@run_package_tests filter = ti -> begin
    filename = get(ti, :filename, nothing)
    filename === nothing && return false
    path = normpath(String(filename))
    return any(p -> startswith(path, p), ALLOWED_PREFIXES)
end
