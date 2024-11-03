import streamlit as st
import matplotlib.pyplot as plt
import random
import numpy as np
from itertools import permutations
import seaborn as sns

# Streamlit app title
st.title("Genetic Algorithm for Travelling Salesman Problem")

# User input for number of cities
num_cities = st.slider("Select number of cities (2-10):", 2, 10, 5)

# User input for coordinates and names of cities
city_coords = {}
for i in range(num_cities):
    city_name = st.text_input(f"Enter name for city {i + 1}:", f"City_{i + 1}")
    x_coord = st.number_input(f"Enter x-coordinate for {city_name}:", min_value=0, max_value=100, value=random.randint(0, 100))
    y_coord = st.number_input(f"Enter y-coordinate for {city_name}:", min_value=0, max_value=100, value=random.randint(0, 100))
    city_coords[city_name] = (x_coord, y_coord)

# Genetic Algorithm Parameters
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Generate pastel colors for cities
colors = sns.color_palette("pastel", len(city_coords))

# Plot cities with connections
fig, ax = plt.subplots()
for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
    color = colors[i]
    ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
    ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='center')

    for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
        if i != j:
            ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

fig.set_size_inches(10, 8)
st.pyplot(fig)

# Initial population
def initial_population(cities_list, n_population=250):
    """
    Generate an initial population by randomly shuffling the list of cities.
    """
    population_perms = []
    for _ in range(n_population):
        individual = cities_list[:]
        random.shuffle(individual)
        population_perms.append(individual)
    return population_perms

# Distance between two cities
def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords)) ** 2))

# Total distance for an individual route
def total_dist_individual(individual):
    total_dist = 0
    for i in range(len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i + 1])
    return total_dist

# Fitness probability function
def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_sum = sum(population_fitness)
    return population_fitness / population_fitness_sum

# Roulette wheel selection
def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    selected_index = np.searchsorted(population_fitness_probs_cumsum, np.random.uniform(0, 1))
    return population[selected_index]

# Crossover
def crossover(parent_1, parent_2):
    cut = round(random.uniform(1, len(parent_1) - 1))
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

# Mutation
def mutation(offspring):
    index_1, index_2 = random.sample(range(len(offspring)), 2)
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

# Run the genetic algorithm
def run_ga(cities_list, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_list, n_population)

    for _ in range(n_generations):
        fitness_probs = fitness_prob(population)
        parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]

        offspring_list = []
        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i + 1])
            if random.random() < mutation_per:
                offspring_1 = mutation(offspring_1)
            if random.random() < mutation_per:
                offspring_2 = mutation(offspring_2)
            offspring_list.extend([offspring_1, offspring_2])

        population = sorted(population + offspring_list, key=total_dist_individual)[:n_population]
    
    return population

# Run GA and find the best path
best_population = run_ga(list(city_coords.keys()), n_population, n_generations, crossover_per, mutation_per)
shortest_path = min(best_population, key=total_dist_individual)
minimum_distance = total_dist_individual(shortest_path)

# Display Results
st.write("### Shortest Path Found")
st.write("Path:", " -> ".join(shortest_path))
st.write("Total Distance Travelled:", round(minimum_distance, 3))

# Plot shortest path
x_shortest = [city_coords[city][0] for city in shortest_path] + [city_coords[shortest_path[0]][0]]
y_shortest = [city_coords[city][1] for city in shortest_path] + [city_coords[shortest_path[0]][1]]

fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i, city in enumerate(shortest_path):
    ax.annotate(f"{i+1}- {city}", (city_coords[city][0], city_coords[city][1]), fontsize=12)

fig.set_size_inches(10, 8)
st.pyplot(fig)
