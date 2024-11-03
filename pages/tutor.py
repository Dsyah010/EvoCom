import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns

st.title("Traveling Salesman Problem using Genetic Algorithm")

# User input for number of cities
n_cities = st.slider("Select the number of cities (between 2 and 10):", min_value=2, max_value=10, value=5)

# User input for city names, coordinates, and icons
city_coords = {}
city_icons = {}

st.subheader("Enter City Details")
for i in range(n_cities):
    name = st.text_input(f"Enter the name of city {i + 1}:", f"City {i+1}")
    x = st.number_input(f"Enter the x coordinate for {name}:", value=float(i * 10))
    y = st.number_input(f"Enter the y coordinate for {name}:", value=float(i * 10))
    icon = st.text_input(f"Enter an icon (emoji or symbol) for {name}:", "🏙️")
    city_coords[name] = (x, y)
    city_icons[name] = icon

# Parameters
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Plotting function for cities
def plot_cities(city_coords, shortest_path=None):
    colors = sns.color_palette("pastel", len(city_coords))
    fig, ax = plt.subplots()
    ax.grid(False)

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons[city]
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                    textcoords='offset points')
        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)
    
    # Plot shortest path if provided
    if shortest_path:
        x_shortest, y_shortest = zip(*[city_coords[city] for city in shortest_path + [shortest_path[0]]])
        ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
        for i, (x, y) in enumerate(zip(x_shortest, y_shortest)):
            ax.annotate(f"{i+1}- {shortest_path[i % len(shortest_path)]}", (x, y), fontsize=15)
        plt.legend()

    fig.set_size_inches(10, 8)
    return fig

# GA Helper Functions
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

def total_dist_individual(individual):
    total_dist = 0
    for i in range(len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1])
    return total_dist

def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(individual) for individual in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_probs = population_fitness / sum(population_fitness)
    return population_fitness_probs

def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    selected_index = np.argmax(population_fitness_probs_cumsum >= np.random.uniform(0, 1))
    return population[selected_index]

def crossover(parent_1, parent_2):
    cut = round(random.uniform(1, len(city_coords) - 1))
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

def mutation(offspring):
    idx1, idx2 = random.sample(range(len(offspring)), 2)
    offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]
    return offspring

def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    for _ in range(n_generations):
        fitness_probs = fitness_prob(population)
        parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]
        offspring_list = []
        for i in range(0, len(parents_list), 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[(i+1) % len(parents_list)])
            if random.random() < mutation_per:
                offspring_1 = mutation(offspring_1)
            if random.random() < mutation_per:
                offspring_2 = mutation(offspring_2)
            offspring_list += [offspring_1, offspring_2]
        population = sorted(parents_list + offspring_list, key=total_dist_individual)[:n_population]
    return population

# Run GA and plot results
if st.button("Run Genetic Algorithm"):
    best_population = run_ga(list(city_coords.keys()), n_population, n_generations, crossover_per, mutation_per)
    shortest_path = min(best_population, key=total_dist_individual)
    minimum_distance = total_dist_individual(shortest_path)
    
    st.write(f"**Shortest Path:** {shortest_path}")
    st.write(f"**Total Distance Traveled:** {round(minimum_distance, 3)}")
    fig = plot_cities(city_coords, shortest_path)
    st.pyplot(fig)
