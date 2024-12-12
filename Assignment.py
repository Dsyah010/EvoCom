import csv
import streamlit as st
import random

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

all_programs = list(program_ratings_dict.keys()) # all programs
all_time_slots = list(range(6, 24)) # time slots

# Crossover
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutating
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

# Fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += program_ratings_dict[program][time_slot]
    return total_rating

# Initializing the population
def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]

    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)

    return all_schedules

# Finding the best schedule
def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0

    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule

    return best_schedule

# Genetic algorithm function
def genetic_algorithm(initial_schedule, generations=GEN, population_size=POP, crossover_rate=0.8, mutation_rate=0.2, elitism_size=EL_S):
    population = [initial_schedule]

    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []

        # Elitism
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        new_population.extend(population[:elitism_size])

        while len(new_population) < population_size:
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

        population = new_population

    return population[0]

# User Interface in Streamlit
st.title("Genetic Algorithm for Program Scheduling")

# Sliders for user input
CO_R = st.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8)  # Default value is 0.8, with range 0 to 0.95
MUT_R = st.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02)  # Default value is 0.02, with range 0.01 to 0.05

# Get all possible schedules and find the best schedule initially
all_possible_schedules = initialize_pop(all_programs, all_time_slots)
initial_best_schedule = finding_best_schedule(all_possible_schedules)

# Calculate the remaining time slots and run the genetic algorithm
rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
genetic_schedule = genetic_algorithm(initial_best_schedule, generations=GEN, population_size=POP, crossover_rate=CO_R, mutation_rate=MUT_R, elitism_size=EL_S)

final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

# Display the final schedule and ratings
st.write("\nFinal Optimal Schedule:")
for time_slot, program in enumerate(final_schedule):
    st.write(f"Time Slot {all_time_slots[time_slot]:02d}:00 - Program {program}")

st.write("Total Ratings:", fitness_function(final_schedule))
