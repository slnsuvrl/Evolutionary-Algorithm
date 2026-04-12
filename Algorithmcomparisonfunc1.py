from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import random
from statistics import mean
import copy
import time

# Function parameters
N = 20               # Number of genes per individual (dimension d = 20)
POP_SIZE = 60        # Population size
MIN_VAL = -10        # Minimum gene value for Function 1
MAX_VAL = 10         # Maximum gene value for Function 1
CROSS_PROB = 0.9     # Crossover probability

class Individual:
    def __init__(self):
        self.gene = [random.uniform(MIN_VAL, MAX_VAL) for _ in range(N)]
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        return (self.gene[0] - 1) ** 2 + sum(
            i * (2 * self.gene[i] ** 2 - self.gene[i - 1]) ** 2 for i in range(1, N)
        )

# Helper functions for GA
def crossover(parent1, parent2):
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    if random.random() < CROSS_PROB:
        point1, point2 = sorted(random.sample(range(N), 2))
        child1.gene[point1:point2], child2.gene[point1:point2] = parent2.gene[point1:point2], parent1.gene[point1:point2]
    child1.fitness = child1.calculate_fitness()
    child2.fitness = child2.calculate_fitness()
    return child1, child2

def mutate(individual, mut_rate, mut_step):
    for i in range(N):
        if random.random() < mut_rate:
            alter = random.uniform(-mut_step, mut_step)
            individual.gene[i] += alter
            individual.gene[i] = max(MIN_VAL, min(MAX_VAL, individual.gene[i]))
    individual.fitness = individual.calculate_fitness()
    return individual

# Genetic Algorithm
def run_ga(gens, mut_rate, mut_step):
    population = [Individual() for _ in range(POP_SIZE)]
    best_individual = min(population, key=lambda ind: ind.fitness)
    
    best_fitness_over_time = []
    mean_fitness_over_time = []

    for gen in range(gens):
        parents = sorted(population, key=lambda ind: ind.fitness)[:POP_SIZE]
        offspring = []
        for i in range(0, POP_SIZE, 2):
            parent1, parent2 = parents[i], parents[min(i+1, POP_SIZE-1)]
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([mutate(child1, mut_rate, mut_step), mutate(child2, mut_rate, mut_step)])
        offspring[random.randint(0, POP_SIZE - 1)] = copy.deepcopy(best_individual)
        population = offspring
        current_best = min(population, key=lambda ind: ind.fitness)
        if current_best.fitness < best_individual.fitness:
            best_individual = copy.deepcopy(current_best)
        best_fitness_over_time.append(best_individual.fitness)
        mean_fitness_over_time.append(mean([ind.fitness for ind in population]))

    return best_individual.fitness, best_fitness_over_time, mean_fitness_over_time

# Simulated Annealing
def run_sa(max_iter, initial_temp, cooling_rate):
    current_solution = Individual()
    best_solution = copy.deepcopy(current_solution)
    temp = initial_temp

    best_fitness_over_time = []

    for i in range(max_iter):
        neighbor = copy.deepcopy(current_solution)
        idx = random.randint(0, N - 1)
        neighbor.gene[idx] += random.uniform(-1, 1)
        neighbor.gene[idx] = max(MIN_VAL, min(MAX_VAL, neighbor.gene[idx]))
        neighbor.fitness = neighbor.calculate_fitness()

        delta_fitness = neighbor.fitness - current_solution.fitness
        if delta_fitness < 0 or random.random() < np.exp(-delta_fitness / temp):
            current_solution = neighbor
            if current_solution.fitness < best_solution.fitness:
                best_solution = copy.deepcopy(current_solution)

        best_fitness_over_time.append(best_solution.fitness)
        temp *= cooling_rate

    return best_solution.fitness, best_fitness_over_time

# Random Hill Climbing
def run_rhc(max_iter):
    current_solution = Individual()
    best_solution = copy.deepcopy(current_solution)

    best_fitness_over_time = []

    for i in range(max_iter):
        neighbor = copy.deepcopy(current_solution)
        idx = random.randint(0, N - 1)
        neighbor.gene[idx] += random.uniform(-1, 1)
        neighbor.gene[idx] = max(MIN_VAL, min(MAX_VAL, neighbor.gene[idx]))
        neighbor.fitness = neighbor.calculate_fitness()

        if neighbor.fitness < current_solution.fitness:
            current_solution = neighbor
            if current_solution.fitness < best_solution.fitness:
                best_solution = copy.deepcopy(current_solution)

        best_fitness_over_time.append(best_solution.fitness)

    return best_solution.fitness, best_fitness_over_time

# Comparison of GA, SA, and RHC
def compare_algorithms():
    print("Running Genetic Algorithm...")
    best_ga_fitness, ga_fitness_plot, _ = run_ga(gens=500, mut_rate=0.6, mut_step=0.2)
    print(f"Best GA Fitness: {best_ga_fitness}")

    print("\nRunning Simulated Annealing...")
    best_sa_fitness, sa_fitness_plot = run_sa(max_iter=500, initial_temp=100, cooling_rate=0.95)
    print(f"Best SA Fitness: {best_sa_fitness}")

    print("\nRunning Random Hill Climbing...")
    best_rhc_fitness, rhc_fitness_plot = run_rhc(max_iter=500)
    print(f"Best RHC Fitness: {best_rhc_fitness}")

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(ga_fitness_plot, label='GA Best Fitness')
    plt.plot(sa_fitness_plot, label='SA Best Fitness', linestyle='dashed')
    plt.plot(rhc_fitness_plot, label='RHC Best Fitness', linestyle='dotted')
    plt.xlabel('Iteration/Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.title('Comparison of GA, SA, and RHC Performance')
    plt.show()

# Run the comparison
compare_algorithms()
