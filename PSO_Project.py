import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import streamlit as st
import os

# Load dataset (the user will upload it or use a fixed path)
st.title("Particle Swarm Optimization for Job Scheduling")

# Define the path to the dataset
file_path = "pages/job_scheduling_data_100.csv"

# Check if the file exists
if os.path.exists(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # Preprocess dataset to extract key parameters
    data['Slack Time'] = data['Due Date'] - (data['Processing Time'] + data['Processing Time'].cumsum())
    data['Machine Utilization'] = 1 - (data['Processing Time'] / data['Processing Time'].sum())
    data = data[['Job ID', 'Slack Time', 'Machine Utilization', 'Processing Time']]

    # Define bounds for optimization
    BOUNDS = {
        "Slack Time": (data['Slack Time'].min(), data['Slack Time'].max()),
        "Machine Utilization": (data['Machine Utilization'].min(), data['Machine Utilization'].max()),
        "Processing Time": (data['Processing Time'].min(), data['Processing Time'].max())
    }

    # Streamlit sliders and inputs for parameter adjustment
    INERTIA = st.slider("Inertia", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    COGNITIVE = st.slider("Cognitive", min_value=0.0, max_value=2.0, value=1.5, step=0.1)
    SOCIAL = st.slider("Social", min_value=0.0, max_value=2.0, value=1.5, step=0.1)
    POP_SIZE = st.slider("Population Size", min_value=10, max_value=200, value=100, step=10)
    GENERATIONS = st.slider("Generations", min_value=10, max_value=500, value=100, step=10)
    TARGET_FITNESS = st.number_input("Target Fitness", min_value=1e5, max_value=1e15, value=1e10, step=1e5)

    # Fitness function (maximize fitness)
    def fitness_cal(position, slack_weight=0.4, utilization_weight=0.3, processing_weight=0.3):
        score = (
            slack_weight * position['Slack Time'] +
            utilization_weight * position['Machine Utilization'] +
            processing_weight * position['Processing Time']
        )
        return score  # Maximizing the score

    # Initialize particles
    def initialize_particles():
        particles = []
        for _ in range(POP_SIZE):
            initial_position = {
                "Slack Time": random.uniform(*BOUNDS["Slack Time"]),
                "Machine Utilization": random.uniform(*BOUNDS["Machine Utilization"]),
                "Processing Time": random.uniform(*BOUNDS["Processing Time"])
            }
            particle = {
                "position": initial_position.copy(),
                "velocity": {
                    "Slack Time": random.uniform(-1, 1),
                    "Machine Utilization": random.uniform(-1, 1),
                    "Processing Time": random.uniform(-1, 1)
                },
                "best_position": initial_position.copy(),
                "best_fitness": float('-inf')
            }
            particles.append(particle)
        return particles

    # Update particle velocity and position
    def update_particle(particle, global_best, inertia, cognitive, social):
        for key in ["Slack Time", "Machine Utilization", "Processing Time"]:
            r1, r2 = random.random(), random.random()
            cognitive_term = cognitive * r1 * (particle["best_position"][key] - particle["position"][key])
            social_term = social * r2 * (global_best[key] - particle["position"][key])
            particle["velocity"][key] = inertia * particle["velocity"][key] + cognitive_term + social_term
            particle["position"][key] += particle["velocity"][key]
            particle["position"][key] = np.clip(particle["position"][key], *BOUNDS[key])  # Enforce bounds

    # Main PSO loop
    def particle_swarm_optimization():
        particles = initialize_particles()
        global_best = None
        global_best_fitness = float('-inf')
        fitness_trends = []

        for generation in range(GENERATIONS):
            for particle in particles:
                # Evaluate fitness
                fitness = fitness_cal(particle["position"])
                if fitness > particle["best_fitness"]:
                    particle["best_fitness"] = fitness
                    particle["best_position"] = particle["position"].copy()

                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best = particle["position"].copy()

            # Update particles
            for particle in particles:
                update_particle(particle, global_best, INERTIA, COGNITIVE, SOCIAL)

            # Log fitness trends
            fitness_trends.append(global_best_fitness)
            st.write(f"Generation {generation + 1}, Best Fitness: {global_best_fitness}")

            # Check stopping criteria
            if global_best_fitness >= TARGET_FITNESS:
                st.write("Optimal solution found!")
                break

        return global_best, fitness_trends

    # Run PSO
    best_solution, fitness_trends = particle_swarm_optimization()

    # Display the best solution
    st.write("Best Solution:", best_solution)

    # Plot fitness trends
    st.subheader("Fitness Trends Over Generations")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fitness_trends, label="Fitness", color='green')
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness Trends Over Generations (Maximization)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

else:
    st.error(f"File not found. Please ensure the file path '{file_path}' is correct.")
