module SteinVariationalGradientDescent

const SVGD = @__MODULE__

export SVGD, SVGDResult, svgd, svgd!, median_heuristic, compute_mean_distance

# using Enzyme: Enzyme, Const, Duplicated
using ProgressLogging: @progress
using Logging: @debug
using Statistics: median
import Zygote
import SimpleKernelRegression
using DispatchDoctor
using ConcreteStructs
# using Enzyme
# import Mooncake # Doesn't work yet with julia 1.12.
import ForwardDiff

"""
    safe_sqrt(x)

A differentiable version of sqrt that handles small negative values by clamping to zero.

This function is useful for numerical stability in gradient computations.
"""
@stable function safe_sqrt(x)
    return sqrt(max(x, zero(x)))
end

"""
    svgd(∇log_p::Function, inits::AbstractMatrix; 
         n_iter=1000, stepsize=0.1, 
         kernel=SimpleKernelRegression.Gaussian(1.0),
         bandwidth_method=:median)

Stein Variational Gradient Descent sampler that approximates a target distribution by 
iteratively updating a set of particles.

# Arguments
- `∇log_p::Function`: gradient of the log probability density function
- `inits::AbstractMatrix`: initial particles matrix of size (n_particles, dimensions)
- `n_iter=1000`: number of iterations
- `stepsize=0.1`: step size for gradient updates
- `kernel=SimpleKernelRegression.Gaussian(1.0)`: Gaussian kernel from SimpleKernelRegression; other kernels are not supported
- `bandwidth_method=:median`: method to set kernel bandwidth. Options: :median, :manual

# Return Value
A named tuple with fields:
- `particles`: final state of the particles
- `distances`: mean pairwise distances between particles at each iteration

# References
Q. Liu and D. Wang, "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm," 
Advances in Neural Information Processing Systems, 2016.
"""
@stable function svgd(
        ∇log_p::Function, inits::AbstractMatrix;
        n_iter = 1000, stepsize = 0.1,
        kernel = SimpleKernelRegression.Gaussian(1.0),
        bandwidth_method = :median,
        progress = true
    )

    kernel isa SimpleKernelRegression.Gaussian || throw(
        ArgumentError("svgd supports only SimpleKernelRegression.Gaussian kernels, got $(typeof(kernel))"),
    )
    particles = copy(inits)
    n_particles, dim = size(particles)

    # Apply median heuristic to set kernel bandwidth if requested
    if bandwidth_method == :median
        h = median_heuristic(particles)
        kernel = SimpleKernelRegression.Gaussian(safe_sqrt(h))
        @info "Using median heuristic bandwidth: $(kernel.σ)"
    end

    # Store initial mean pairwise distance for convergence monitoring
    distances = zeros(n_iter + 1)
    distances[1] = compute_mean_distance(particles)

    # Main SVGD loop
    @info "Running SVGD for $n_iter iterations with $n_particles particles"
    temp = similar(particles)

    if progress
        @progress for i in 1:n_iter
            @inbounds begin
                # Update all particles
                for j in 1:n_particles
                    temp[j, :] = update_particle(view(particles, j, :), particles, ∇log_p, kernel, stepsize)
                end

                # Copy updated particles
                particles .= temp

                # Fast-fail on NaN divergence
                if any(isnan, particles)
                    error("NaN detected in particles at iteration $i. Algorithm diverged. Check stepsize, kernel bandwidth, or gradient function.")
                end

                # Store mean pairwise distance for monitoring
                distances[i + 1] = compute_mean_distance(particles)
            end
        end
    else
        for i in 1:n_iter
            @inbounds begin
                # Update all particles
                for j in 1:n_particles
                    temp[j, :] = update_particle(view(particles, j, :), particles, ∇log_p, kernel, stepsize)
                end

                # Copy updated particles
                particles .= temp

                # Fast-fail on NaN divergence
                if any(isnan, particles)
                    error("NaN detected in particles at iteration $i. Algorithm diverged. Check stepsize, kernel bandwidth, or gradient function.")
                end

                # Store mean pairwise distance for monitoring
                distances[i + 1] = compute_mean_distance(particles)
            end
        end
    end

    return (particles = particles, distances = distances)
end

"""
Compute the median heuristic for setting kernel bandwidth.
Returns median squared distance / log(n+1)
"""
@stable function median_heuristic(particles)
    n_particles = size(particles, 1)

    # Handle edge case: single particle
    if n_particles <= 1
        return 0.0
    end

    # Use SimpleKernelRegression's pDist2Squared function to compute pairwise distances
    dist_matrix = SimpleKernelRegression.pDist2Squared(particles, particles)

    # Extract upper triangular part (excluding diagonal)
    dists = [dist_matrix[i, j] for i in 1:n_particles for j in (i + 1):n_particles]

    # Return median heuristic
    return median(dists) / log(n_particles + 1)
end

"""
Calculate the mean pairwise distance between particles
"""
@stable function compute_mean_distance(particles::AbstractArray{T}) where {T <: Real}
    n_particles = size(particles, 1)

    # Handle edge case: single particle
    if n_particles <= 1
        return zero(T)
    end

    dist_matrix = safe_sqrt.(SimpleKernelRegression.pDist2Squared(particles, particles))

    # Calculate mean of upper triangular part (excluding diagonal)
    total_dist = sum(dist_matrix[i, j] for i in 1:n_particles for j in (i + 1):n_particles)
    count = n_particles * (n_particles - 1) / 2
    return total_dist / count
end

@stable function update_particle(particle, all_particles, ∇log_p, kernel::SimpleKernelRegression.Gaussian, stepsize)
    n_particles, dim = size(all_particles)

    # Materialize particle view to avoid closure issues
    particle_vec = Vector{eltype(particle)}(particle)
    update = zeros(eltype(particle), dim)

    # Compute kernel between two points (ForwardDiff-compatible, avoids sqrt)
    function kernel_between_points(x1, x2)
        ElemType = promote_type(eltype(x1), eltype(x2))
        dist_sq = zero(ElemType)
        @inbounds for d in 1:length(x1)
            dist_sq += (x1[d] - x2[d])^2
        end
        # Direct Gaussian kernel: K(x1,x2) = exp(-||x1-x2||²/(2σ²))
        # Avoids sqrt to prevent NaN in gradient at zero distance
        return exp(-dist_sq / (2 * kernel.σ^2))
    end

    # Preallocate buffer for particle j
    xj_buffer = Vector{eltype(particle)}(undef, dim)

    @inbounds for j in 1:n_particles
        # Copy particle j to buffer
        for d in 1:dim
            xj_buffer[d] = all_particles[j, d]
        end

        # Compute kernel value
        k_val = kernel_between_points(xj_buffer, particle_vec)

        # First term: K(x_j, x_i) ∇log_p(x_j)
        update .+= k_val .* ∇log_p(xj_buffer)

        # Second term: ∇_{x_j} K(x_j, x_i)
        # Compute gradient w.r.t. xj (the first argument) using ForwardDiff
        grad = ForwardDiff.gradient(xj -> kernel_between_points(xj, particle_vec), xj_buffer)

        # Add the gradient
        update .+= grad
    end

    # Apply step size and return updated particle
    update .*= (stepsize / n_particles)
    return particle_vec .+ update
end


end # module
