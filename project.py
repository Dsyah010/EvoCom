import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Streamlit page configuration
st.set_page_config(page_title="Ant Colony Optimization for Job Scheduling", layout="wide")

# Title
st.title("Ant Colony Optimization for Job Scheduling")

# Sidebar for parameters
st.sidebar.header("ACO Parameters")
NUM_ANTS = st.sidebar.slider("Number of Ants", 10, 100, 50)
NUM_ITERATIONS = st.sidebar.slider("Number of Iterations", 10, 500, 100)
ALPHA = st.sidebar.slider("Pheromone Importance (α)", 0.1, 5.0, 1.0)
BETA = st.sidebar.slider("Heuristic Importance (β)", 0.1, 5.0, 2.0)
EVAPORATION_RATE = st.sidebar.slider("Evaporation Rate", 0.01, 1.0, 0.5, 0.01)
Q = st.sidebar.slider("Pheromone Deposit Factor (Q)", 10, 500, 100)

# Sidebar for custom weights
st.sidebar.header("Custom Weights")
w1 = st.sidebar.slider("Lateness Weight (w1)", 0.0, 1.0, 0.5, 0.1)
w2 = st.sidebar.slider("Utilization Weight (w2)", 0.0, 1.0, 0.3, 0.1)
w3 = st.sidebar.slider("Priority Weight (w3)", 0.0, 1.0, 0.2, 0.1)

# Load the dataset from the local file
try:
    dataset_path = "pages/job_scheduling_data_100.csv"
    df = pd.read_csv(dataset_path)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Bounds for discretization
    bounds = {
        "processing_time": (df["Processing Time"].min(), df["Processing Time"].max()),
        "queue_length": (df["Queue Length"].min(), df["Queue Length"].max()),
        "machine_utilization": (df["Machine Utilization"].min(), df["Machine Utilization"].max()),
    }

    # Initialize pheromones
    def initialize_pheromones():
        pheromones = {
            "processing_time": np.ones(int(bounds["processing_time"][1] - bounds["processing_time"][0] + 1)),
            "queue_length": np.ones(int(bounds["queue_length"][1] - bounds["queue_length"][0] + 1)),
            "machine_utilization": np.ones(int(bounds["machine_utilization"][1] - bounds["machine_utilization"][0] + 1)),
        }
        return pheromones

    # Fitness function
    def fitness_cal(solution):
        lateness = max(0, (solution["arrival_time"] + solution["processing_time"] + solution["setup_time"] - solution["due_date"]))
        utilization = solution["machine_utilization"]
        priority = solution["job_priority"]
        return w1 * (-lateness) + w2 * utilization + w3 * priority

    # Main ACO loop
    def ant_colony_optimization(data):
        pheromones = initialize_pheromones()
        best_solution = None
        best_fitness = float('-inf')
        fitness_trends = []

        for iteration in range(NUM_ITERATIONS):
            solutions = []
            fitness_values = []

            for _ in range(NUM_ANTS):
                row = data.iloc[random.randint(0, len(data) - 1)]
                solution = {
                    "processing_time": row["Processing Time"],
                    "queue_length": row["Queue Length"],
                    "machine_utilization": row["Machine Utilization"],
                    "arrival_time": row["Arrival Time"],
                    "due_date": row["Due Date"],
                    "setup_time": row["Setup Time"],
                    "job_priority": row["Job Priority"],
                }
                fitness = fitness_cal(solution)
                solutions.append(solution)
                fitness_values.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution

            # Update pheromones
            for solution, fitness in zip(solutions, fitness_values):
                pheromones["processing_time"][solution["processing_time"] - bounds["processing_time"][0]] += Q / (-fitness)
                pheromones["queue_length"][solution["queue_length"] - bounds["queue_length"][0]] += Q / (-fitness)
                utilization_index = int((solution["machine_utilization"] - bounds["machine_utilization"][0]) / (bounds["machine_utilization"][1] - bounds["machine_utilization"][0]))
                utilization_index = np.clip(utilization_index, 0, len(pheromones["machine_utilization"]) - 1)
                pheromones["machine_utilization"][utilization_index] += Q / (-fitness)

            for key in pheromones:
                pheromones[key] *= (1 - EVAPORATION_RATE)

            fitness_trends.append(best_fitness)
            st.text(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")

        return best_solution, fitness_trends

    # Run ACO
    if st.button("Run ACO"):
        with st.spinner("Running Ant Colony Optimization..."):
            best_solution, fitness_trends = ant_colony_optimization(df)

        st.subheader("Best Solution")
        st.write(best_solution)

        st.subheader("Fitness Trends Over Iterations")
        plt.figure(figsize=(12, 6))
        plt.plot(fitness_trends, label="Fitness")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.title("Fitness Trends Over Iterations")
        plt.legend()
        st.pyplot(plt)

except Exception as e:
    st.error(f"Failed to load dataset. Error: {e}")
