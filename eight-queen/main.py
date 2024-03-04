import numpy as np
import matplotlib.pyplot as plt
import random
import time

N_QUEENS = 8
MAX_SCORE = 28  # Max pairs of queens not attacking each other

def fitness(chromosome):
    non_attacking = 0
    for i in range(len(chromosome)):
        for j in range(i+1, len(chromosome)):
            if (chromosome[i] != chromosome[j]) and (abs(i - j) != abs(chromosome[i] - chromosome[j])):
                non_attacking += 1
    return non_attacking

def random_chromosome(size):
    return [random.randint(1, size) for _ in range(size)]

def tournament_selection(population, scores, k=3):
    selection_ix = np.random.randint(len(population))
    for ix in np.random.randint(0, len(population), k-1):
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return population[selection_ix]

def crossover(p1, p2, crossover_rate=0.9):
    if random.random() < crossover_rate:
        point = random.randint(1, N_QUEENS-2)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]
    return p1, p2

def mutation(chromosome, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx = random.randint(0, N_QUEENS-1)
        chromosome[idx] = random.randint(1, N_QUEENS)
    return chromosome

def genetic_algorithm(n_generations=100000, population_size=100, crossover_rate=0.9, mutation_rate=0.01, elitism=True):
    population = [random_chromosome(N_QUEENS) for _ in range(population_size)]
    best_score_progress = []  # Track the best score
    avg_score_progress = []  # Track the average score

    best_individual = None
    best_individual_score = 0

    for generation in range(n_generations):
        scores = [fitness(chromo) for chromo in population]
        current_best_score = max(scores)
        avg_score = sum(scores) / population_size
        best_score_progress.append(current_best_score)
        avg_score_progress.append(avg_score)

        if best_individual is None or current_best_score > best_individual_score:
            best_individual = population[scores.index(current_best_score)]
            best_individual_score = current_best_score

        if current_best_score == MAX_SCORE: break  # Stop if we find a perfect solution

        selected = [tournament_selection(population, scores) for _ in range(population_size)]
        new_population = list()
        for i in range(0, population_size, 2):
            p1, p2 = selected[i], selected[i+1]
            offspring1, offspring2 = crossover(p1, p2, crossover_rate)
            new_population.append(mutation(offspring1, mutation_rate))
            if len(new_population) < population_size:
                new_population.append(mutation(offspring2, mutation_rate))
                
        if elitism:
            worst_index = scores.index(min(scores))
            new_population[worst_index] = best_individual
        
        population = new_population
    
    return best_score_progress, avg_score_progress, best_individual


def random_search(n_attempts=10000):
    best_solution = None
    best_score = 0
    
    for attempt in range(n_attempts):
        chromosome = random_chromosome(N_QUEENS)
        score = fitness(chromosome)
        if score > best_score:
            best_score = score
            best_solution = chromosome
        if best_score == MAX_SCORE: break
    
    return best_score, best_solution

time_GA_Start = 0
time_GA_End = 0
time_R_Start = 0
time_R_End = 0

time_GA_Start = time.time()
ga_best_score_progress, ga_avg_score_progress, ga_best_solution = genetic_algorithm(n_generations=100, population_size=100)
time_GA_End = time.time()

time_R_Start = time.time()
random_best_score, random_best_solution = random_search()
time_R_End = time.time()

print(ga_best_solution)

print("Time for Random: ", time_R_End - time_R_Start, "Best fitness: ", fitness(random_best_solution))
print("Time for GA: ", time_GA_End - time_GA_Start, "GA Best fitness: ", fitness(ga_best_solution))

plt.figure(figsize=(12, 6))

# Best Score Progress
plt.plot(ga_best_score_progress, label='Best Score')

# Average Score Progress
plt.plot(ga_avg_score_progress, label='Average Score', linestyle='--')

plt.title('Learning Curve: Genetic Algorithm for Eight Queens Puzzle')
plt.xlabel('Generation')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()