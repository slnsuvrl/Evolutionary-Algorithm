from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import random
from statistics import mean
import copy
import time

# Constants specific to Function 2
N = 20               # Number of genes per individual (dimension d = 20)
POP_SIZE = 100        # Population size 
MIN_VAL = -500       # Minimum gene value
MAX_VAL = 500        # Maximum gene value
YOU = 61             # The last two digits of the student number

# Genetic Algorithm-specific parameters for grid search
GENS_RANGE = [350, 450, 500 ]  
MUT_RATES = [0.2, 0.4, 0.6]      
MUT_STEPS = [ 0.6, 0.7, 0.8]   
CROSS_PROB = 0.9             

# Process time starts
time_start = time.time()

class Individual:
    def __init__(self):
        self.gene = [random.uniform(MIN_VAL, MAX_VAL) for _ in range(N)]
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        # f(x) = 418.9829 * d - sum(x_i * sin(sqrt(|x_i|))) for i=1 to d + YOU
        sum_term = sum(self.gene[i] * np.sin(np.sqrt(abs(self.gene[i]))) for i in range(N))
        return ((418.9829 * N) - sum_term )+ YOU


# Initialize population
def initialize_population():
    return [Individual() for _ in range(POP_SIZE)]

# Selection function (simple elitism-based)
def elitism_selection(population):
    # Select the best individuals to form the next generation
    return sorted(population, key=lambda ind: ind.fitness)[:POP_SIZE]

# Two-point crossover
def crossover(parent1, parent2):
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    if random.random() < CROSS_PROB:
        point1, point2 = sorted(random.sample(range(N), 2))
        child1.gene[point1:point2], child2.gene[point1:point2] = parent2.gene[point1:point2], parent1.gene[point1:point2]
    child1.fitness = child1.calculate_fitness()
    child2.fitness = child2.calculate_fitness()
    return child1, child2

# Mutation function
def mutate(individual, mut_rate, mut_step):
    for i in range(N):
        if random.random() < mut_rate:
            alter = random.uniform(-mut_step, mut_step)
            individual.gene[i] += alter
            # Clamp the gene to the defined limits
            individual.gene[i] = max(MIN_VAL, min(MAX_VAL, individual.gene[i]))
    individual.fitness = individual.calculate_fitness()
    return individual

# Genetic Algorithm
def run_ga(gens, mut_rate, mut_step):
    population = initialize_population()
    best_individual = min(population, key=lambda ind: ind.fitness)
    
    # Lists for storing fitness values to plot later
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

# Grid Search for Hyperparameter Tuning
def grid_search():
    results = []
    best_overall_fitness = float('inf')
    best_params = None
    best_fitness_over_time = None
    mean_fitness_over_time = None

    # Data to be plotted in the 3D scatter plot
    gens_list = []
    mut_rate_list = []
    mut_step_list = []
    fitness_list = []

    for gens in GENS_RANGE:
        for mut_rate in MUT_RATES:
            for mut_step in MUT_STEPS:
                print(f"Testing GENS={gens}, MUT_RATE={mut_rate}, MUT_STEP={mut_step}")
                best_fitness, best_fitness_over_time, mean_fitness_over_time = run_ga(gens, mut_rate, mut_step)

                # Print the best fitness value for the current configuration
                print(f"Best Fitness for GENS={gens}, MUT_RATE={mut_rate}, MUT_STEP={mut_step}: {best_fitness}")
                
                # Record results for this configuration
                results.append((gens, mut_rate, mut_step, best_fitness))

                # Store the best fitness for plotting
                gens_list.append(gens)
                mut_rate_list.append(mut_rate)
                mut_step_list.append(mut_step)
                fitness_list.append(best_fitness)
                
                # Update best parameters if this configuration is the best so far
                if best_fitness < best_overall_fitness:
                    best_overall_fitness = best_fitness
                    best_params = (gens, mut_rate, mut_step)
                    best_fitness_plot = best_fitness_over_time
                    mean_fitness_plot = mean_fitness_over_time
    
    # Display the best parameters and corresponding fitness
    print("\nBest Hyperparameters Found:")
    print(f"GENS={best_params[0]}, MUT_RATE={best_params[1]}, MUT_STEP={best_params[2]}")
    print(f"Best Fitness: {best_overall_fitness}")
    # Process time ends
    time_end = time.time()
    print("Process time:", time_end - time_start)

    # Plotting: 3D scatter plot of results
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the scatter plot and store it in a variable
    scatter = ax.scatter(gens_list, mut_rate_list, mut_step_list, c=fitness_list, cmap='viridis', s=60)
    
    # Set axis labels and title
    ax.set_xlabel('Number of Generations (GENS)')
    ax.set_ylabel('Mutation Rate (MUT_RATE)')
    ax.set_zlabel('Mutation Step (MUT_STEP)')
    ax.set_title('3D Scatter Plot of Best Fitness Values')

    # Set axis ranges
    ax.set_xlim([min(gens_list), max(gens_list)])
    ax.set_ylim([min(mut_rate_list), max(mut_rate_list)])
    ax.set_zlim([min(mut_step_list), max(mut_step_list)])

    # Color bar to show fitness values
    cbar = plt.colorbar(scatter)
    cbar.set_label('Fitness Value')

    # Show the plot
    plt.show()

    

    # General plot for fitness over generations for the best configuration
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_plot, label='Best Fitness')
    plt.plot(mean_fitness_plot, label='Mean Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.title(f'Best Fitness Over Generations (GENS={best_params[0]}, MUT_RATE={best_params[1]}, MUT_STEP={best_params[2]})')
    plt.show()



# Run the Grid Search
grid_search()
