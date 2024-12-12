import csv
import streamlit as st
import random
import pandas as pd

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        # Skip the header
        header = next(reader)

        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings

    return program_ratings

# Path to the CSV file
file_path = 'pages/program_ratings.csv'

# Get the data in the required format
program_ratings_dict = read_csv_to_dict(file_path)

# Defining parameters for the genetic algorithm
GEN = 100
POP = 50
EL_S = 2

all_programs = list(program_ratings_dict.keys())  # all programs
all_time_slots = list(range(6, 24))  # time slots

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutating
def mutate(schedule):
    if not schedule:
        return schedule
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if time_slot < len(program_ratings_dict[program]):
            total_rating += program_ratings_dict[program][time_slot]
    return total_rating

# Initializing the population
def initialize_pop(programs, time_slots, population_size):
    population = []
    for _ in range(population_size):
        schedule = random.sample(programs, len(programs))
        population.append(schedule)
    return population

# Genetic algorithm function
def genetic_algorithm(initial_population, generations=GEN, crossover_rate=0.8, mutation_rate=0.02, elitism_size=EL_S):
    population = initial_population

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < len(population):
            parent1, parent2 = random.choices(population, k=2)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1)
            if random.random() < mutation_rate:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = new_population[:len(population)]

    return population[0]

# User Interface in Streamlit
st.title("Genetic Algorithm for Program Scheduling")

# Sliders for user input
CO_R = st.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8)  # Default value is 0.8, range 0 to 0.95
MUT_R = st.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02)  # Default value is 0.02, range 0.01 to 0.05

# Initialize population
initial_population = initialize_pop(all_programs, all_time_slots, POP)

# Run the genetic algorithm
optimal_schedule = genetic_algorithm(initial_population, generations=GEN, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S)

# Ensure the schedule matches the number of time slots
if len(optimal_schedule) < len(all_time_slots):
    # Pad with a default value (e.g., a random program)
    optimal_schedule += [random.choice(all_programs) for _ in range(len(all_time_slots) - len(optimal_schedule))]
elif len(optimal_schedule) > len(all_time_slots):
    # Truncate to match the time slots
    optimal_schedule = optimal_schedule[:len(all_time_slots)]

# Prepare the final schedule for display in a table
schedule_data = {
    "Time Slot": [f"{hour}:00" for hour in all_time_slots],
    "Scheduled Program": optimal_schedule
}

# Convert the data to a pandas DataFrame for better table display
df_schedule = pd.DataFrame(schedule_data)

# Display the final schedule as a table
st.write("**Final Optimal Schedule:**")
st.table(df_schedule)

# Display total ratings
st.write("Total Ratings:", fitness_function(optimal_schedule))
