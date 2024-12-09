import csv
import random
import streamlit as st

# Function to read the CSV file and convert it to the desired format
def read_csv_to_dict(file_path):
    program_ratings = {}
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip the header
        for row in reader:
            program = row[0]
            ratings = [float(x) for x in row[1:]]  # Convert the ratings to floats
            program_ratings[program] = ratings
    return program_ratings

# Path to the CSV file
file_path = 'pages/program_ratings.csv'
program_ratings_dict = read_csv_to_dict(file_path)

# Defining parameters and dataset
GEN = 100
POP = 50
EL_S = 2

all_programs = list(program_ratings_dict.keys())  # All programs
all_time_slots = list(range(6, 24))  # Time slots
ratings = program_ratings_dict

# Defining functions
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        total_rating += ratings[program][time_slot]
    return total_rating

def initialize_pop(programs, time_slots):
    if not programs:
        return [[]]
    all_schedules = []
    for i in range(len(programs)):
        for schedule in initialize_pop(programs[:i] + programs[i + 1:], time_slots):
            all_schedules.append([programs[i]] + schedule)
    return all_schedules

def finding_best_schedule(all_schedules):
    best_schedule = []
    max_ratings = 0
    for schedule in all_schedules:
        total_ratings = fitness_function(schedule)
        if total_ratings > max_ratings:
            max_ratings = total_ratings
            best_schedule = schedule
    return best_schedule

def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[mutation_point] = new_program
    return schedule

def genetic_algorithm(initial_schedule, generations, population_size, crossover_rate, mutation_rate, elitism_size):
    population = [initial_schedule]
    for _ in range(population_size - 1):
        random_schedule = initial_schedule.copy()
        random.shuffle(random_schedule)
        population.append(random_schedule)

    for generation in range(generations):
        new_population = []
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

# Streamlit interface
st.title("Genetic Algorithm for Optimal Program Scheduling")

# User inputs for the genetic algorithm
CO_R = st.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8, step=0.01)
MUT_R = st.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.2, step=0.01)

# Run the algorithm
all_possible_schedules = initialize_pop(all_programs, all_time_slots)
initial_best_schedule = finding_best_schedule(all_possible_schedules)
rem_t_slots = len(all_time_slots) - len(initial_best_schedule)
genetic_schedule = genetic_algorithm(
    initial_best_schedule,
    generations=GEN,
    population_size=POP,
    crossover_rate=CO_R,
    mutation_rate=MUT_R,
    elitism_size=EL_S,
)
final_schedule = initial_best_schedule + genetic_schedule[:rem_t_slots]

# Prepare data for the table
schedule_table = {
    "Time Slot": [f"{time_slot:02d}:00" for time_slot in all_time_slots],
    "Program": final_schedule,
}

# Display the results
st.subheader("Final Optimal Schedule (Table Format)")
st.dataframe(schedule_table)  # Interactive table
# Alternatively, use st.table(schedule_table) for a static table

# Display total ratings
st.write("Total Ratings:", fitness_function(final_schedule))
