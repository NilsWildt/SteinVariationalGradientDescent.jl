#!/usr/bin/env julia

using SimpleKernelRegression
using Random
using Statistics
using SteinVariationalGradientDescent
using UnicodePlots

rng = Random.MersenneTwister(2024)
μ = [0.5, -1.0]
Σ = [1.0 0.3; 0.3 0.5]

∇log_p(x) = -(Σ \ (x .- μ))

initial_particles = randn(rng, 400, 2) .* 3
result = svgd(∇log_p, initial_particles;
              n_iter = 500,
              stepsize = 0.2,
              bandwidth_method = :median,
              kernel = SimpleKernelRegression.Gaussian(1.0),
              progress = true)

final_particles = result.particles

println("\n=== Summary ===")
println("Initial mean: ", vec(mean(initial_particles, dims=1)))
println("Final mean:   ", vec(mean(final_particles, dims=1)))
println("Target mean:  ", μ)
println("Final distance trace: first = $(result.distances[1]), last = $(result.distances[end])\n")

initial_plot = scatterplot(initial_particles[:, 1], initial_particles[:, 2];
                           title = "Initial particles",
                           xlabel = "x₁",
                           ylabel = "x₂",
                           width = 40,
                           height = 20,
                           color = :light_blue)

final_plot = scatterplot(final_particles[:, 1], final_particles[:, 2];
                         title = "Stein variational particles",
                         xlabel = "x₁",
                         ylabel = "x₂",
                         width = 40,
                         height = 20,
                         color = :green)

println(initial_plot)
println()
println(final_plot)
