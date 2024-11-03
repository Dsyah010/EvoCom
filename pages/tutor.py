import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import seaborn as sns

# Prompt user for number of cities
while True:
    try:
        n_cities = int(input("Enter the number of cities (between 2 and 10): "))
        if 2 <= n_cities <= 10:
            break
        else:
            print("Please enter a number between 2 and 10.")
    except ValueError:
        print("Invalid input. Please enter a number between 2 and 10.")

# Input city details
city_coords = {}
city_icons = {}

for i in range(n_cities):
    name = input(f"Enter the name of city {i + 1}: ")
    x = float(input(f"Enter the x coordinate for {name}: "))
    y = float(input(f"Enter the y coordinate for {name}: "))
    icon = input(f"Enter an icon (emoji or symbol) for {name}: ")
    city_coords[name] = (x, y)
    city_icons[name] = icon

# Algorithm parameters
n_population = 250
crossover_per = 0.8
mutation_per = 0.2
n_generations = 200

# Plot cities
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

fig.set_size_inches(16, 12)
plt.show()

# Initialize population
def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    # Ensure n_population is not larger than the number of possible permutations
    n_population = min(n_population, len(possible_perms))  
    random_ids = random.sample(range(0, len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

# Distance calculation
def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

# Total distance for a path
def total_dist_individual(individual):
    total_dist = 0
    for i in range(len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1])
    return total_dist

# Fitness function
def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(individual) for individual in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_probs = population_fitness / sum(population_fitness)
    return population_fitness_probs

# Roulette wheel selection
def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    selected_index = np.argmax(population_fitness_probs_cumsum >= np.random.uniform(0, 1))
    return population[selected_index]

# Crossover function
def crossover(parent_1, parent_2):
    cut = round(random.uniform(1, len(city_coords) - 1))
    offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
    offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
    return offspring_1, offspring_2

# Mutation function
def mutation(offspring):
    idx1, idx2 = random.sample(range(len(offspring)), 2)
    offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]
    return offspring

# Genetic Algorithm
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

# Run GA and find best path
best_population = run_ga(list(city_coords.keys()), n_population, n_generations, crossover_per, mutation_per)
shortest_path = min(best_population, key=total_dist_individual)
minimum_distance = total_dist_individual(shortest_path)

# Plotting the shortest path
x_shortest, y_shortest = zip(*[city_coords[city] for city in shortest_path + [shortest_path[0]]])
fig, ax = plt.subplots()
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
plt.legend()

for i, (x, y) in enumerate(zip(x_shortest, y_shortest)):
    ax.annotate(f"{i+1}- {shortest_path[i % len(shortest_path)]}", (x, y), fontsize=20)

plt.title(f"TSP Best Route Using GA\nTotal Distance Travelled: {round(minimum_distance, 3)}")
plt.suptitle(f"{n_generations} Generations\n{n_population} Population Size\n{crossover_per} Crossover\n{mutation_per} Mutation", fontsize=12, y=1.02)
fig.set_size_inches(16, 12)
plt.show()
