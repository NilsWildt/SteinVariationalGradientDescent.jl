#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.instantiate()

using CairoMakie
using Colors
# using DifferentiationInterface
using LinearAlgebra: norm, det
using Printf: @sprintf
using Random
using SimpleKernelRegression
using Statistics
using SteinVariationalGradientDescent

CairoMakie.activate!(; type = "png")

rng = Random.MersenneTwister(20240708)
μ = [0.5, -1.0]
Σ = [1.0 0.3; 0.3 0.5]

∇log_p(x) = -(Σ \ (x .- μ))

initial_particles = randn(rng, 128, 2) .* 3
kernel = SimpleKernelRegression.Gaussian(0.5)
result = svgd(∇log_p, initial_particles;
              n_iter = 1024,
              stepsize = 0.1,
              bandwidth_method = :median,
              kernel = kernel, progress=true)

final_particles = result.particles

println("\n=== Summary ===")
println("Initial mean: ", vec(mean(initial_particles, dims = 1)))
println("Final mean:   ", vec(mean(final_particles, dims = 1)))
println("Target mean:  ", μ)
println("Final distance trace: first = $(result.distances[1]), last = $(result.distances[end])\n")

bg = RGBAf(0.98, 0.98, 0.99, 1.0)
initial_color = RGBf(0.349, 0.537, 0.839)
final_color = RGBf(0.188, 0.521, 0.247)
target_color = RGBf(0.835, 0.196, 0.160)

const INV_SQRT_2PI = inv(sqrt(2π))

function kde1d(xs::AbstractVector{<:Real}, grid::AbstractVector{<:Real})
    n = length(xs)
    n == 0 && return zeros(Float64, length(grid))
    σ_emp = 1.06 * std(xs; corrected = false)
    σ = σ_emp > eps(real(σ_emp)) ? σ_emp * n^(-1 / 5) : one(Float64)
    inv_σ = inv(σ)
    scale = INV_SQRT_2PI * inv_σ / n
    dens = zeros(Float64, length(grid))
    for (i, xg) in enumerate(grid)
        acc = 0.0
        for x in xs
            z = (xg - x) * inv_σ
            acc += exp(-0.5 * z * z)
        end
        dens[i] = scale * acc
    end
    return dens
end

normal_pdf(x, μ_val, σ_val) = INV_SQRT_2PI / σ_val * exp(-0.5 * ((x - μ_val) / σ_val)^2)

Σ_inv = inv(Matrix(Σ))
Σ_det = det(Matrix(Σ))
norm_2d = 1 / (2π * sqrt(Σ_det))

gaussian_pdf(x, y) = begin
    dx = x - μ[1]
    dy = y - μ[2]
    quad = Σ_inv[1, 1] * dx * dx + 2 * Σ_inv[1, 2] * dx * dy + Σ_inv[2, 2] * dy * dy
    norm_2d * exp(-0.5 * quad)
end

all_particles = vcat(initial_particles, final_particles)
x_values = all_particles[:, 1]
y_values = all_particles[:, 2]
x_span = maximum(x_values) - minimum(x_values)
y_span = maximum(y_values) - minimum(y_values)
pad_x = max(0.15 * x_span, 0.5)
pad_y = max(0.15 * y_span, 0.5)
x_min, x_max = minimum(x_values) - pad_x, maximum(x_values) + pad_x
y_min, y_max = minimum(y_values) - pad_y, maximum(y_values) + pad_y

x_grid = LinRange(x_min, x_max, 400)
y_grid = LinRange(y_min, y_max, 400)
target_x = normal_pdf.(x_grid, μ[1], sqrt(Σ[1, 1]))
target_y = normal_pdf.(y_grid, μ[2], sqrt(Σ[2, 2]))
initial_kde_x = kde1d(view(initial_particles, :, 1), x_grid)
initial_kde_y = kde1d(view(initial_particles, :, 2), y_grid)
final_kde_x = kde1d(view(final_particles, :, 1), x_grid)
final_kde_y = kde1d(view(final_particles, :, 2), y_grid)

density_vals = [gaussian_pdf(x, y) for x in x_grid, y in y_grid]

fig = Figure(; size = (1200, 900), fontsize = 22, backgroundcolor = bg)

ax_top = Axis(fig[1, 1];
              xlabel = "",
              ylabel = "density",
              title = "Marginal density of x₁",
              xticksvisible = false,
              xticklabelsvisible = false,
              backgroundcolor = bg)

ax_joint = Axis(fig[2, 1];
                xlabel = "x₁",
                ylabel = "x₂",
                title = "SVGD trajectory",
                aspect = DataAspect(),
                backgroundcolor = bg)

ax_right = Axis(fig[2, 2];
                xlabel = "density",
                ylabel = "",
                title = "Marginal density of x₂",
                yaxisposition = :right,
                backgroundcolor = bg)

mean_error = norm(vec(mean(final_particles, dims = 1)) - μ)
distance_drop = result.distances[1] - result.distances[end]
stats_text = @sprintf "Mean error: %.3f\nDistance drop: %.3f\nFinal distance: %.3f" mean_error distance_drop result.distances[end]

fig[1, 2] = Label(fig,
                  stats_text;
                  fontsize = 20,
                  color = RGBAf(0.15, 0.15, 0.16, 0.85),
                  halign = :left,
                  valign = :top,
                  padding = (16, 16, 16, 16),
                  tellwidth = false,
                  tellheight = false)

# Set layout sizes after all axes are created
colgap!(fig.layout, 16)
rowgap!(fig.layout, 16)
rowsize!(fig.layout, 1, Relative(0.3))
rowsize!(fig.layout, 2, Relative(0.7))
colsize!(fig.layout, 1, Relative(0.7))
colsize!(fig.layout, 2, Relative(0.3))

initial_alpha = RGBAf(initial_color.r, initial_color.g, initial_color.b, 0.35)
final_alpha = RGBAf(final_color.r, final_color.g, final_color.b, 0.9)
target_alpha = RGBAf(target_color.r, target_color.g, target_color.b, 0.85)

lines!(ax_top, x_grid, initial_kde_x; color = initial_alpha, linewidth = 2.0)
lines!(ax_top, x_grid, final_kde_x; color = final_alpha, linewidth = 2.6)
lines!(ax_top, x_grid, target_x; color = target_color, linewidth = 2.4, linestyle = :dash)

contour!(ax_joint, x_grid, y_grid, density_vals;
         levels = 7,
         linewidth = 1.5,
         color = RGBAf(target_color.r, target_color.g, target_color.b, 0.55),
         linestyle = :dash,
         label = "Target density")
scatter!(ax_joint, initial_particles[:, 1], initial_particles[:, 2];
         color = initial_alpha,
         markersize = 6,
         strokewidth = 0.0,
         label = "Initial")
scatter!(ax_joint, final_particles[:, 1], final_particles[:, 2];
         color = final_alpha,
         markersize = 6,
         strokewidth = 0.0,
         label = "Final")
scatter!(ax_joint, [μ[1]], [μ[2]];
         color = target_alpha,
         marker = :star4,
         markersize = 18,
         strokewidth = 0.0,
         label = "Target mean")

lines!(ax_right, initial_kde_y, y_grid; color = initial_alpha, linewidth = 2.0)
lines!(ax_right, final_kde_y, y_grid; color = final_alpha, linewidth = 2.6)
lines!(ax_right, target_y, y_grid; color = target_color, linewidth = 2.4, linestyle = :dash)

limits!(ax_joint, x_min, x_max, y_min, y_max)
limits!(ax_top, x_min, x_max, nothing, nothing)
limits!(ax_right, nothing, nothing, y_min, y_max)

hidexdecorations!(ax_top; grid = false)
hideydecorations!(ax_right; grid = false)
linkxaxes!(ax_joint, ax_top)
linkyaxes!(ax_joint, ax_right)

axislegend(ax_joint; position = :rb, orientation = :vertical, framevisible = false, padding = 4)

Label(fig[0, 1:2],
      "SteinVariationalGradientDescent.jl × CairoMakie.jl",
      fontsize = 28,
      color = final_color,
      halign = :center,
      font = :bold)

Label(fig[2, 1:2],
      "1200 particles, 600 SVGD iterations, median bandwidth",
      fontsize = 18,
      color = RGBA(0.2, 0.2, 0.2, 0.75),
      halign = :center)

output_path = joinpath(@__DIR__, "gaussian_svgd.png")
save(output_path, fig; px_per_unit = 2.0)
@info "Saved figure" output_path
