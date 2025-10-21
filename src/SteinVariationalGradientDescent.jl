module SteinVariationalGradientDescent

const SVGD = @__MODULE__

export SVGD, SVGDResult, svgd, svgd!, median_heuristic, compute_mean_distance

# using Enzyme: Enzyme, Const, Duplicated
using ProgressLogging: @progress
using Logging: @debug
using Statistics: median
import Zygote
import SimpleKernelRegression
using DifferentiationInterface
using DispatchDoctor
using ConcreteStructs
# const ENZYME_REVERSE = Enzyme.set_runtime_activity(Enzyme.Reverse)

"""
    struct SVGDResult

Container holding the particles and diagnostic distances produced by [`svgd`](@ref).
"""
@concrete struct SVGDResult{A<:AbstractMatrix{<:Real}, V<:AbstractVector{<:Real}}
    particles::A
    distances::V
end

function Base.show(io::IO, result::SVGDResult)
    n_particles, dim = size(result.particles)
    print(io, "SVGDResult with $n_particles particles in $dim dimensions")
end

"""
    safe_sqrt(x)

Return `sqrt(max(x, zero(x)))` to avoid spurious complex values from round-off.
"""
@inline function safe_sqrt(x)
    return sqrt(max(x, zero(x)))
end

"""
    svgd(∇log_p, inits; kwargs...)

Run Stein Variational Gradient Descent starting from `inits` and return an [`SVGDResult`](@ref).
The gradient callback `∇log_p` must accept a particle (as `AbstractVector`) and return a vector
compatible with the particle type.
"""
@stable function svgd(∇log_p, inits::AbstractMatrix{T}; kwargs...) where {T<:Real}
    particles = copy(inits)
    return svgd!(particles, ∇log_p; kwargs...)
end

"""
    svgd!(particles, ∇log_p; n_iter=1_000, stepsize=0.1, kernel=nothing,
          bandwidth_method=:median, progress=true)

In-place variant that overwrites `particles` with the SVGD iterate. Returns an [`SVGDResult`](@ref)
whose `particles` field aliases the provided storage, enabling zero-copy workflows.
"""
function svgd!(particles::AbstractMatrix{T}, ∇log_p;
               n_iter::Integer = 1_000,
               stepsize::Real = 0.1,
               kernel::Union{SimpleKernelRegression.SKernel, Nothing} = nothing,
               bandwidth_method::Symbol = :median,
               progress::Bool = true) where {T<:Real}
    T <: AbstractFloat ||
        throw(ArgumentError("particles must have floating-point eltype; got $T"))
    n_iter ≥ 0 || throw(ArgumentError("n_iter must be non-negative, got $n_iter"))
    n_particles = size(particles, 1)
    n_particles > 0 || throw(ArgumentError("particles must contain at least one particle"))
    dim = size(particles, 2)

    S = float(T)
    stepsize_T = convert(S, stepsize)
    inv_n_particles = one(S) / S(n_particles)

    kernel_obj = prepare_kernel(kernel, bandwidth_method, particles, S)
    distances = Vector{S}(undef, n_iter + 1)
    distances[1] = compute_mean_distance(particles)

    temp = similar(particles)
    particle_buffer = zeros(S, dim)
    dest_buffer = zeros(S, dim)
    update_buffer = zeros(S, dim)
    kernel_grad_buffer = zeros(S, dim)

    if progress
        @progress for iter in 1:n_iter
            iterate_svgd!(temp, particles, ∇log_p, kernel_obj, stepsize_T,
                          inv_n_particles, particle_buffer, dest_buffer,
                          update_buffer, kernel_grad_buffer)
            distances[iter + 1] = compute_mean_distance(particles)
        end
    else
        for iter in 1:n_iter
            iterate_svgd!(temp, particles, ∇log_p, kernel_obj, stepsize_T,
                          inv_n_particles, particle_buffer, dest_buffer,
                          update_buffer, kernel_grad_buffer)
            distances[iter + 1] = compute_mean_distance(particles)
        end
    end

    return SVGDResult(particles, distances)
end

"""
    median_heuristic(particles)

Median heuristic for bandwidth selection:

``σ² = median(‖xᵢ - xⱼ‖²) / log(n_particles + 1)``.
"""
@stable function median_heuristic(particles::AbstractMatrix{T}) where {T<:Real}
    n_particles = size(particles, 1)
    n_particles > 1 || return zero(float(T))

    dist_matrix = SimpleKernelRegression.pDist2Squared(particles, particles)
    S = float(eltype(dist_matrix))
    count = n_particles * (n_particles - 1) ÷ 2
    work = Vector{S}(undef, count)

    idx = 1
    @inbounds for j in 2:n_particles
        for i in 1:j-1
            work[idx] = dist_matrix[i, j]
            idx += 1
        end
    end

    return median(work) / log(S(n_particles + 1))
end

"""
    compute_mean_distance(particles)

Compute the mean pairwise (Euclidean) distance across the particle ensemble.
"""
function compute_mean_distance(particles::AbstractMatrix{T}) where {T<:Real}
    n_particles = size(particles, 1)
    n_particles > 1 || return zero(float(T))

    dist_matrix = SimpleKernelRegression.pDist2Squared(particles, particles)
    total = zero(float(T))
    count = 0

    @inbounds for j in 2:n_particles
        for i in 1:j-1
            total += safe_sqrt(dist_matrix[i, j])
            count += 1
        end
    end

    return total / count
end

@inline function iterate_svgd!(temp, particles, ∇log_p, kernel, stepsize, inv_n_particles,
                               particle_buffer, dest_buffer, update_buffer, kernel_grad_buffer)
    n_particles = size(particles, 1)

    @inbounds for row in 1:n_particles
        @views copyto!(particle_buffer, particles[row, :])
        update_particle!(dest_buffer, particle_buffer, particles, ∇log_p, kernel,
                         stepsize, inv_n_particles, update_buffer, kernel_grad_buffer)
        @views copyto!(temp[row, :], dest_buffer)
    end

    copyto!(particles, temp)

    return nothing
end

function update_particle!(dest::AbstractVector{T}, particle::AbstractVector{T},
                          all_particles::AbstractMatrix, ∇log_p, kernel,
                          stepsize::T, inv_n_particles::T,
                          update_buffer::AbstractVector{T},
                          kernel_grad_buffer::AbstractVector{T}) where {T<:Real}
    copyto!(dest, particle)
    fill!(update_buffer, zero(T))

    @inbounds for row in 1:size(all_particles, 1)
        xj = view(all_particles, row, :)
        grad_j = ∇log_p(xj)
        k_val = kernel_value_gradient!(kernel_grad_buffer, kernel, xj, particle)
        k_coeff = convert(T, k_val)

        @inbounds for idx in eachindex(update_buffer)
            update_buffer[idx] = muladd(k_coeff, convert(T, grad_j[idx]), update_buffer[idx])
            update_buffer[idx] += kernel_grad_buffer[idx]
        end
    end

    scale = stepsize * inv_n_particles
    @inbounds for idx in eachindex(dest)
        dest[idx] += scale * update_buffer[idx]
    end

    return dest
end

@inline function kernel_value_gradient!(grad_buffer::AbstractVector{T},
                                        kernel::SimpleKernelRegression.Gaussian,
                                        xj, particle::AbstractVector{T}) where {T<:Real}
    σ = convert(T, kernel.σ)
    σ² = σ * σ
    inv_σ² = inv(σ²)

    dist_sq = zero(T)
    @inbounds for idx in eachindex(grad_buffer)
        diff = convert(T, xj[idx]) - particle[idx]
        dist_sq += diff * diff
        grad_buffer[idx] = diff
    end

    k_val = exp(-dist_sq * (inv_σ² / 2))
    coeff = k_val * inv_σ²

    @inbounds for idx in eachindex(grad_buffer)
        grad_buffer[idx] *= coeff
    end

    return k_val
end

function kernel_value_gradient!(grad_buffer::AbstractVector{T},
                                kernel::SimpleKernelRegression.SKernel,
                                xj, particle::AbstractVector{T}) where {T<:Real}
    function kernel_between_points(x1, x2)
        dist_sq = zero(T)
        @inbounds for idx in eachindex(x1)
            diff = convert(T, x1[idx]) - x2[idx]
            dist_sq += diff * diff
        end
        return convert(T, SimpleKernelRegression.evalKernel(kernel, safe_sqrt(dist_sq)))
    end

    value = kernel_between_points(xj, particle)
    # Enzyme.make_zero!(grad_buffer)
    # Enzyme.autodiff(ENZYME_REVERSE, Const(kernel_between_points), Const(xj),
                    # Duplicated(particle, grad_buffer))
    grad_buffer .= DifferentiationInterface.gradient(x -> kernel_between_points(x, particle), xj)[1]
    return value
end

function prepare_kernel(kernel::Union{SimpleKernelRegression.SKernel, Nothing},
                        method::Symbol,
                        particles,
                        ::Type{S}) where {S<:Real}
    method ∈ (:median, :manual) ||
        throw(ArgumentError("Unsupported bandwidth_method $method"))

    kernel_obj = if kernel === nothing
        SimpleKernelRegression.Gaussian(one(S))
    else
        kernel
    end

    if method == :manual
        return kernel_obj
    end

    kernel_obj isa SimpleKernelRegression.Gaussian ||
        throw(ArgumentError("Median heuristic is only implemented for Gaussian kernels"))

    h = median_heuristic(particles)
    σ = safe_sqrt(convert(S, h))
    σ = σ > zero(S) ? σ : one(S)
    @debug "Using median heuristic bandwidth" σ = σ
    return SimpleKernelRegression.Gaussian(σ)
end

end # module SteinVariationalGradientDescent
