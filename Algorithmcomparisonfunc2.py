from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
from statistics import mean

# Constants specific to Function 2
N = 20               # Number of genes per individual (dimension d = 20)
POP_SIZE = 100        # Population size 
MIN_VAL = -500       # Minimum gene value
MAX_VAL = 500        # Maximum gene value
YOU = 61             # The last two digits of the student number

# Common parameters for all algorithms
GENS = 500           # Number of generations
MUT_RATE = 0.4       # Mutation rate for GA and RHC
MUT_STEP = 0.7       # Mutation step for GA and RHC
INITIAL_TEMP = 1000  # Initial temperature for SA
COOLING_RATE = 0.99  # Cooling rate for SA
CROSS_PROB = 0.9     # Crossover probability for GA

# Process time starts
time_start = time.time()

class Individual:
    def __init__(self):
        self.gene = [random.uniform(MIN_VAL, MAX_VAL) for _ in range(N)]
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        # f(x) = 418.9829 * d - sum(x_i * sin(sqrt(|x_i|))) for i=1 to d + YOU
        sum_term = sum(self.gene[i] * np.sin(np.sqrt(abs(self.gene[i]))) for i in range(N))
        return ((418.9829 * N) - sum_term ) + YOU

# Initialize population for GA
def initialize_population():
    return [Individual() for _ in range(POP_SIZE)]

# Selection function (simple elitism-based)
def elitism_selection(population):
    return sorted(population, key=lambda ind: ind.fitness)[:POP_SIZE]

# Two-point crossover for GA
def crossover(parent1, parent2):
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    if random.random() < CROSS_PROB:
        point1, point2 = sorted(random.sample(range(N), 2))
        child1.gene[point1:point2], child2.gene[point1:point2] = parent2.gene[point1:point2], parent1.gene[point1:point2]
    child1.fitness = child1.calculate_fitness()
    child2.fitness = child2.calculate_fitness()
    return child1, child2

# Mutation function for GA and RHC
def mutate(individual, mut_rate, mut_step):
    for i in range(N):
        if random.random() < mut_rate:
            alter = random.uniform(-mut_step, mut_step)
            individual.gene[i] += alter
            individual.gene[i] = max(MIN_VAL, min(MAX_VAL, individual.gene[i]))
    individual.fitness = individual.calculate_fitness()
    return individual

# Genetic Algorithm (GA)
def run_ga(gens, mut_rate, mut_step):
    population = initialize_population()
    best_individual = min(population, key=lambda ind: ind.fitness)
    
    # Lists for tracking fitness over time
    best_fitness_over_time = []
    mean_fitness_over_time = []

    for gen in range(gens):
        # Select parents using elitism
        parents = elitism_selection(population)
        
        # Crossover and mutation to produce offspring
        offspring = []
        for i in range(0, POP_SIZE, 2):
            parent1, parent2 = parents[i], parents[min(i+1, POP_SIZE-1)]
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([mutate(child1, mut_rate, mut_step), mutate(child2, mut_rate, mut_step)])

        # Elitism: Preserve the best individual from the previous generation
        offspring[random.randint(0, POP_SIZE - 1)] = copy.deepcopy(best_individual)
        
        # Update population and track the best individual
        population = offspring
        current_best = min(population, key=lambda ind: ind.fitness)
        if current_best.fitness < best_individual.fitness:
            best_individual = copy.deepcopy(current_best)
        
        # Record statistics for this generation
        best_fitness_over_time.append(best_individual.fitness)
        mean_fitness_over_time.append(mean([ind.fitness for ind in population]))

    return best_individual.fitness, best_fitness_over_time, mean_fitness_over_time

# Simulated Annealing (SA)
def run_sa(gens, initial_temp, cooling_rate):
    current_solution = Individual()
    best_solution = current_solution
    
    # Lists for tracking fitness over time
    best_fitness_over_time = []
    mean_fitness_over_time = []
    
    for gen in range(gens):
        temp = initial_temp * (cooling_rate ** gen)
        new_solution = Individual()
        
        # Calculate fitness difference
        delta_fitness = current_solution.fitness - new_solution.fitness
        
        # Acceptance probability
        if delta_fitness > 0 or random.random() < np.exp(delta_fitness / temp):
            current_solution = new_solution
            if current_solution.fitness < best_solution.fitness:
                best_solution = current_solution
        
        # Record statistics
        best_fitness_over_time.append(best_solution.fitness)
        mean_fitness_over_time.append(current_solution.fitness)
    
    return best_solution.fitness, best_fitness_over_time, mean_fitness_over_time

# Random Hill Climbing (RHC)
def run_rhc(gens, mut_rate, mut_step):
    current_solution = Individual()
    best_solution = current_solution

    # Lists for tracking fitness over time
    best_fitness_over_time = []
    mean_fitness_over_time = []
    
    for gen in range(gens):
        new_solution = copy.deepcopy(current_solution)
        
        # Random mutation
        for i in range(N):
            if random.random() < mut_rate:  # Mutation with a set rate
                new_solution.gene[i] += random.uniform(-mut_step, mut_step)
                new_solution.gene[i] = max(MIN_VAL, min(MAX_VAL, new_solution.gene[i]))
        
        new_solution.fitness = new_solution.calculate_fitness()
        
        # If the new solution is better, replace the current solution
        if new_solution.fitness < current_solution.fitness:
            current_solution = new_solution
            if current_solution.fitness < best_solution.fitness:
                best_solution = current_solution
        
        # Record statistics
        best_fitness_over_time.append(best_solution.fitness)
        mean_fitness_over_time.append(current_solution.fitness)
    
    return best_solution.fitness, best_fitness_over_time, mean_fitness_over_time

# Run all algorithms and compare results
def compare_algorithms():
    print("Running Genetic Algorithm (GA)...")
    ga_fitness, ga_best_fitness_over_time, ga_mean_fitness_over_time = run_ga(GENS, MUT_RATE, MUT_STEP)
    print(f"GA Best Fitness: {ga_fitness}")
    
    print("Running Simulated Annealing (SA)...")
    sa_fitness, sa_best_fitness_over_time, sa_mean_fitness_over_time = run_sa(GENS, INITIAL_TEMP, COOLING_RATE)
    print(f"SA Best Fitness: {sa_fitness}")
    
    print("Running Random Hill Climbing (RHC)...")
    rhc_fitness, rhc_best_fitness_over_time, rhc_mean_fitness_over_time = run_rhc(GENS, MUT_RATE, MUT_STEP)
    print(f"RHC Best Fitness: {rhc_fitness}")

    # Plotting fitness over generations for all algorithms
    plt.figure(figsize=(12, 6))
    plt.plot(ga_best_fitness_over_time, label='GA Best Fitness')
    plt.plot(sa_best_fitness_over_time, label='SA Best Fitness')
    plt.plot(rhc_best_fitness_over_time, label='RHC Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Comparison of Algorithms Over Generations')
    plt.legend()
    plt.show()

# Run the comparison
compare_algorithms()

# Process time ends
time_end = time.time()
print("Process time:", time_end - time_start)
