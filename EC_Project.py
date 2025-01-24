import streamlit as st
import pandas as pd
import numpy as np
import random
from gplearn.genetic import SymbolicRegressor
import graphviz

# Constants
TARGET_FITNESS = -97850

# Load the dataset
DATA_PATH = "EvoCom/pages/job_scheduling_data_100.csv"
data = pd.read_csv(DATA_PATH)

# App Header
st.title("Job Scheduling Optimization using Genetic Algorithm and Symbolic Regression")

# Dataset Display
if st.checkbox("Show Dataset"):
    st.write(data.head())

# Initialization Function
def initialize_pop(pop_size):
    population = []
    for _ in range(pop_size):
        individual = {
            "slack_time": random.randint(1, 50),
            "machine_utilization": random.uniform(1, 20),
            "processing_time": random.randint(50, 100),
        }
        population.append(individual)
    return population

# Fitness Calculation Function
def fitness_cal(individual):
    slack_time = individual["slack_time"]
    machine_utilization = individual["machine_utilization"]
    processing_time = individual["processing_time"]

    cost = (
        processing_time * 1000 +
        machine_utilization * 1500 +
        slack_time * 20
    )

    efficiency = min(100, (
        machine_utilization * 0.3 + processing_time * 0.2
    ))

    return -cost + efficiency - slack_time

# Selection Function
def selection(population):
    fitness_scores = [(ind, fitness_cal(ind)) for ind in population]
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    selected = [ind[0] for ind in fitness_scores[:len(population) // 2]]
    return selected

# Crossover Function
def crossover(parent1, parent2):
    child = {
        "slack_time": random.choice([parent1["slack_time"], parent2["slack_time"]]),
        "machine_utilization": random.choice([parent1["machine_utilization"], parent2["machine_utilization"]]),
        "processing_time": random.choice([parent1["processing_time"], parent2["processing_time"]]),
    }
    return child

# Mutation Function
def mutate(individual, mut_rate):
    if random.random() < mut_rate:
        individual["slack_time"] = random.randint(1, 50)
    if random.random() < mut_rate:
        individual["machine_utilization"] = random.uniform(1, 20)
    if random.random() < mut_rate:
        individual["processing_time"] = random.randint(50, 500)
    return individual

# Genetic Algorithm Main Function
def genetic_algorithm(pop_size, mut_rate):
    population = initialize_pop(pop_size)
    generation = 1
    best_fitness_values = []

    while True:
        selected = selection(population)
        new_generation = []
        for _ in range(pop_size):
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = crossover(parent1, parent2)
            child = mutate(child, mut_rate)
            new_generation.append(child)
        population = new_generation

        best_individual = max(population, key=lambda ind: fitness_cal(ind))
        best_fitness = fitness_cal(best_individual)
        best_fitness_values.append(best_fitness)

        st.write(f"Generation {generation}, Best Fitness: {best_fitness}")

        if best_fitness >= TARGET_FITNESS:
            st.success("Optimal solution found!")
            break

        generation += 1

    return population, best_fitness_values

# Streamlit User Inputs
st.sidebar.header("Genetic Algorithm Parameters")
user_pop_size = st.sidebar.slider("Population Size", 10, 200, 100)
user_mut_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2, 0.01)

# Run Genetic Algorithm
if st.button("Run Genetic Algorithm"):
    st.write("Running Genetic Algorithm...")
    _, fitness_values = genetic_algorithm(user_pop_size, user_mut_rate)
    st.line_chart(fitness_values)

# Symbolic Regression
st.sidebar.header("Symbolic Regression Parameters")
gp_population = st.sidebar.slider("Population Size", 100, 1000, 500)
gp_generations = st.sidebar.slider("Generations", 10, 50, 20)

if st.button("Run Symbolic Regression"):
    # Extract features and targets
    X = data[[
        "Processing Time", "Setup Time", "Queue Length", "Slack Time", "Machine Utilization"
    ]].values

    y = []
    for _, row in data.iterrows():
        fitness = fitness_cal({
            "slack_time": row["Slack Time"],
            "machine_utilization": row["Machine Utilization"],
            "processing_time": row["Processing Time"]
        })
        y.append(fitness)
    y = np.array(y)

    # Train symbolic regression model
    gp = SymbolicRegressor(
        population_size=gp_population,
        generations=gp_generations,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=1,
        random_state=42
    )
    gp.fit(X, y)

    # Display results
    st.write("Evolved Expression:")
    st.text(gp._program)

    # Visualize the symbolic regression tree
    dot_data = gp._program.export_graphviz()
    st.graphviz_chart(dot_data)
