import csv
import random
import streamlit as st
import pandas as pd

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

# Fitness function
def fitness_function(schedule):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if program in ratings and time_slot < len(ratings[program]):
            total_rating += ratings[program][time_slot]
        else:
            total_rating += 0  # Assign default value for invalid cases
    
    # Debugging: Show fitness calculation
    st.write(f"Schedule: {schedule}")
    st.write(f"Fitness: {total_rating}")
    
    return total_rating

# Initialize population
def initialize_pop(programs, time_slots):
    population = []
    for _ in range(POP):  # Generate a population of schedules
        random_schedule = random.sample(programs, len(programs))
        random_schedule = random_schedule[:len(time_slots)]  # Trim to match time slots
        population.append(random_schedule)
    
    # Debugging: Print first few schedules
    st.write("Initial Population (First 5 Schedules):")
    for i, schedule in enumerate(population[:5]):
        st.write(f"Schedule {i+1}: {schedule}")
    
    return population

# Crossover function
def crossover(schedule1, schedule2):
    crossover_point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

# Mutation function
def mutate(schedule):
    mutation_point = random.randint(0, len(schedule) - 1)
    new_program = random.choice([p for p in all_programs if p != schedule[mutation_point]])
    schedule[mutation_point] = new_program
    return schedule

# Genetic algorithm
def genetic_algorithm(generations, population_size, crossover_rate, mutation_rate, elitism_size):
    population = initialize_pop(all_programs, all_time_slots)
    
    for generation in range(generations):
        # Sort by fitness and keep the best schedules
        population.sort(key=lambda schedule: fitness_function(schedule), reverse=True)
        
        # Debugging: Display the best schedule for the current generation
        st.write(f"Generation {generation+1}: Best Fitness = {fitness_function(population[0])}")
        st.write(f"Best Schedule: {population[0]}")
        
        new_population = population[:elitism_size]  # Elitism

        # Generate new offspring
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
        
        population = new_population[:population_size]
    
    # Final debugging: Display the best schedule before post-processing
    st.write("Final Best Schedule (Raw):", population[0])
    return population[0]

# Streamlit interface
st.title("Genetic Algorithm for Optimal Program Scheduling")

# User inputs for the genetic algorithm
CO_R = st.sidebar.slider("Crossover Rate (CO_R)", 0.0, 0.95, 0.8, step=0.01)
MUT_R = st.sidebar.slider("Mutation Rate (MUT_R)", 0.01, 0.05, 0.02, step=0.01)
if st.sidebar.button("Run Algorithm"):
    # Run the algorithm
    best_schedule = genetic_algorithm(
        generations=GEN,
        population_size=POP,
        crossover_rate=CO_R,
        mutation_rate=MUT_R,
        elitism_size=EL_S,
    )

    # Debugging: Length checks
    st.write("Length of best schedule:", len(best_schedule))
    st.write("Length of time slots:", len(all_time_slots))

    # Adjust the length of the schedule to match the time slots
    if len(best_schedule) < len(all_time_slots):
        st.warning("Schedule is shorter than time slots. Adding 'None' to fill.")
        best_schedule.extend(["None"] * (len(all_time_slots) - len(best_schedule)))
    elif len(best_schedule) > len(all_time_slots):
        st.warning("Schedule is longer than time slots. Trimming.")
        best_schedule = best_schedule[:len(all_time_slots)]

    # Prepare data for the table
    schedule_data = {
        "Time Slot": [f"{time_slot:02d}:00" for time_slot in all_time_slots],
        "Program": best_schedule,
    }
    vertical_table = pd.DataFrame(schedule_data)

    # Debugging: Check table structure
    st.write("Data for Table:", vertical_table)

    # Display the table
    st.subheader("Final Optimal Schedule (Vertical Format)")
    st.table(vertical_table)  # Display the DataFrame as a static table
    st.write("Total Ratings:", fitness_function(best_schedule))
