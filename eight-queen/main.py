import numpy as np
import random
import time


# Parameters
population_size = 100
crossover_rate = 0.8
mutation_rate = 0.01
generations = 100

# Fitness Function
def fitness(chromosome):
    # Counts the non-attacking pairs of queens
    non_attacking = 0
    for i in range(len(chromosome)):
        for j in range(i+1, len(chromosome)):
            if (chromosome[i] != chromosome[j]) and (abs(i - j) != abs(chromosome[i] - chromosome[j])):
                non_attacking += 1
    return non_attacking

# Selection Function: Tournament Selection
def selection(population):
    tournament_size = 5
    selected = random.sample(population, tournament_size)
    selected = sorted(selected, key=lambda x: fitness(x), reverse=True)
    return selected[0]

# Crossover Function: One-Point Crossover
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1)-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return [child1, child2]
    else:
        return [parent1, parent2]

# Mutation Function
def mutate(chromosome):
    if random.random() < mutation_rate:
        index = random.randint(0, len(chromosome)-1)
        chromosome[index] = random.randint(0, len(chromosome)-1)
    return chromosome

# Generate Initial Population
def generate_population(size):
    return [random.sample(range(8), 8) for _ in range(size)]

# Genetic Algorithm
def genetic_algorithm():
    population = generate_population(population_size)
    best_solution = None
    best_fitness = 0
    
    for generation in range(generations):
        new_population = []
        for _ in range(int(len(population)/2)):
            parent1 = selection(population)
            parent2 = selection(population)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])
        
        # Evaluate
        population = new_population
        current_best = max(population, key=lambda x: fitness(x))
        current_best_fitness = fitness(current_best)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best
        
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
        
        # Stop if solution is found
        if best_fitness == 28:
            break
    
    return best_solution, best_fitness

# Random Search for Comparison
def random_search(tries=10000):
    best_solution = None
    best_fitness = 0
    for _ in range(tries):
        solution = random.sample(range(8), 8)
        current_fitness = fitness(solution)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_solution = solution
        if best_fitness == 28:
            break
    return best_solution, best_fitness

# Run Genetic Algorithm
start_GA = time.time()
ga_solution, ga_fitness = genetic_algorithm()
end_GA = time.time()
print("time for GA: ", end_GA-start_GA)
print(f"GA Solution: {ga_solution} with fitness {ga_fitness}")

# Run Random Search
start_R = time.time()
rs_solution, rs_fitness = random_search()
end_R = time.time()
print("time for RANDOM: ", end_R-start_R)
print(f"Random Search Solution: {rs_solution} with fitness {rs_fitness}")
