using Plots
using StatsBase
using Random

# Define linear dynamical system model
function dynamics(x_prev, t)
    A = 0.9   # State transition coefficient
    B = 1.0   # Control input coefficient
    u = sin(t)   # Control input (sine function for example)
    x_new = A * x_prev + B * u + randn() * 0.1   # Add noise
    return x_new
end

# Particle filter
function particle_filter(measurements, num_particles)
    num_steps = length(measurements)

    particles_history = []
    particles = randn(num_particles)  # Initialize particles randomly

    for t in 1:num_steps
        # Prediction
        if t > 1
            particles = [dynamics(p, t) for p in particles]
        end

        # Update
        weights = [pdf(Normal(p, 0.5), measurements[t]) for p in particles]
        weights /= sum(weights)

        # Resample
        indices = sample(1:num_particles, Weights(weights), num_particles)
        particles = particles[indices]

        push!(particles_history, copy(particles))
    end

    return particles_history
end

# Generate synthetic data
function generate_data(num_steps, initial_state)
    true_states = zeros(num_steps)
    measurements = zeros(num_steps)

    true_states[1] = initial_state
    measurements[1] = initial_state + randn() * 0.5  # Initial measurement

    for t in 2:num_steps
        true_states[t] = dynamics(true_states[t-1], t)
        measurements[t] = true_states[t] + randn() * 0.5
    end

    return true_states, measurements
end

# Main function to generate synthetic data and run particle filter
function main(num_steps, num_particles, initial_state)
    true_states, measurements = generate_data(num_steps, initial_state)
    particles_history = particle_filter(measurements, num_particles)

    return true_states, measurements, particles_history
end

# Plot the particles
function plot_particles(particles, true_state, measurement)
    plot(title="Particle Filter", xlabel="Time", ylabel="Value", legend=:topright)
    plot!(particles, color=:black, markersize=1, markerstrokecolor=:black, markerstrokewidth=0, label="Particles")
    plot!([true_state], color=:blue, marker=:circle, markersize=5, label="True State")
    plot!([measurement], color=:red, marker=:square, markersize=5, label="Measurement")
end

# Generate synthetic data and run particle filter
Random.seed!(123) # For reproducibility
num_steps = 50
num_particles = 100
initial_state = 0.0

true_states, measurements, particles_history = main(num_steps, num_particles, initial_state)

# Create GIF using @animate macro

guess = zeros(100)
for t in 1:50
    guess[t] = mean(particles_history[t])
end

anim = @animate for (t, particles) in enumerate(particles_history)
    y = particles_history[t]
    x = ones(100) * t
    v = collect(zip(x, y))
    p = plot([true_states[1:t], guess[1:t]], color=[:red :dodgerblue2], label=["Truth" "Estimate"], legend=:outertopright)
    scatter!(p, v, color=:dodgerblue2, label="particles")
end
gif(anim, "particle_filter.gif", fps=3)

