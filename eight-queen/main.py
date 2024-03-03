import numpy as np

def initialize_population(size=100):
    return np.random.randint(0, 8, (size, 8))

def fitness(chromosome):
    conflicts = 0
    for i in range(len(chromosome)):
        for j in range(i + 1, len(chromosome)):
            if chromosome[i] == chromosome[j] or abs(chromosome[i] - chromosome[j]) == j - i:
                conflicts += 1
    return 28 - conflicts

def select(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        participants = np.random.choice(len(population), tournament_size, replace=False)
        selected.append(population[max(participants, key=lambda idx: fitnesses[idx])])
    return np.array(selected)

def crossover(parent1, parent2, crossover_rate=0.9):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1) - 1)
        return np.concatenate([parent1[:point], parent2[point:]]), np.concatenate([parent2[:point], parent1[point:]])
    else:
        return parent1, parent2

def mutate(chromosome, mutation_rate=0.01):
    if np.random.rand() < mutation_rate:
        idx = np.random.randint(len(chromosome))
        chromosome[idx] = np.random.randint(8)
    return chromosome

def genetic_algorithm():
    population_size = 100
    generations = 1000
    population = initialize_population(population_size)
    
    for generation in range(generations):
        fitnesses = np.array([fitness(chromosome) for chromosome in population])
        if 28 in fitnesses:
            print("Solution found:", population[np.argmax(fitnesses)])
            break
        selected = select(population, fitnesses)
        offspring = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected[i], selected[i+1]
            child1, child2 = crossover(parent1, parent2)
            offspring.append(mutate(child1))
            offspring.append(mutate(child2))
        population = np.array(offspring)

genetic_algorithm()
