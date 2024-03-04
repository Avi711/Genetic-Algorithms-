import numpy as np
import random
import itertools
import time
import matplotlib.pyplot as plt


def load_coordinates(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    coordinates = [tuple(map(float, line.strip().split())) for line in lines]
    return coordinates

def calculate_distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def create_distance_matrix(coordinates):
    num_cities = len(coordinates)
    distance_matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distance_matrix[i, j] = calculate_distance(coordinates[i], coordinates[j])
    return distance_matrix

def generate_chromosome(num_cities):
    return random.sample(range(1, num_cities + 1), num_cities)

def fitness(chromosome, distance_matrix):
    total_distance = 0
    num_cities = len(chromosome)
    for i in range(num_cities - 1):
        total_distance += distance_matrix[chromosome[i] - 1, chromosome[i + 1] - 1]
    total_distance += distance_matrix[chromosome[-1] - 1, chromosome[0] - 1] 
    return total_distance

def tournament_selection(population, tournament_size, distance_matrix):
    selected = random.sample(population, tournament_size)
    return min(selected, key=lambda x: fitness(x, distance_matrix))

def ordered_crossover(parent1, parent2):
    start = random.randint(0, len(parent1) - 1)
    end = random.randint(start + 1, len(parent1))
    child = [-1] * len(parent1)

    child[start:end] = parent1[start:end]

    remaining_genes = [gene for gene in parent2 if gene not in child]
    idx = 0
    for i in range(len(parent1)):
        if child[i] == -1:
            child[i] = remaining_genes[idx]
            idx += 1

    return child

def swap_mutation(chromosome):
    idx1, idx2 = random.sample(range(len(chromosome)), 2)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

# def genetic_algorithm(num_cities, population_size, generations, tournament_size, crossover_rate, mutation_rate, distance_matrix):
#     population = [generate_chromosome(num_cities) for _ in range(population_size)]

#     for generation in range(generations):
#         new_population = []
#         for _ in range(population_size // 2):
#             parent1 = tournament_selection(population, tournament_size, distance_matrix)
#             parent2 = tournament_selection(population, tournament_size, distance_matrix)
#             child1 = ordered_crossover(parent1, parent2)
#             child2 = ordered_crossover(parent2, parent1)
#             child1 = swap_mutation(child1)
#             child2 = swap_mutation(child2)
#             new_population.extend([child1, child2])

#         population = new_population

#         # Print the best fitness in each generation
#         best_solution = min(population, key=lambda x: fitness(x, distance_matrix))
#         print(f"Generation {generation}: Best Fitness = {fitness(best_solution, distance_matrix)}")

#    
#     return min(population, key=lambda x: fitness(x, distance_matrix))




def genetic_algorithm(num_cities, population_size, generations, tournament_size, crossover_rate, mutation_rate, distance_matrix):
    population = [generate_chromosome(num_cities) for _ in range(population_size)]
    best_fitness_over_generations = []
    average_fitness_over_generations = []

    for generation in range(generations):
        print(generation)
        new_population = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, tournament_size, distance_matrix)
            parent2 = tournament_selection(population, tournament_size, distance_matrix)
            if random.random() < crossover_rate:
                child1 = ordered_crossover(parent1, parent2)
                child2 = ordered_crossover(parent2, parent1)
            else:
                child1, child2 = parent1, parent2

            if random.random() < mutation_rate:
                child1 = swap_mutation(child1)
            if random.random() < mutation_rate:
                child2 = swap_mutation(child2)

            new_population.extend([child1, child2])

        population = new_population

        fitness_values = [fitness(chromosome, distance_matrix) for chromosome in population]
        best_fitness = min(fitness_values)
        average_fitness = sum(fitness_values) / len(fitness_values)
        best_fitness_over_generations.append(best_fitness)
        average_fitness_over_generations.append(average_fitness)

    best_solution = min(population, key=lambda x: fitness(x, distance_matrix))
    return best_solution, best_fitness_over_generations, average_fitness_over_generations

def greedy_algorithm(distance_matrix):
    num_cities = len(distance_matrix)
    unvisited_cities = set(range(2, num_cities + 1))
    current_city = 1
    tour = [current_city]

    while unvisited_cities:
        closest_city = min(unvisited_cities, key=lambda x: distance_matrix[current_city - 1, x - 1])
        tour.append(closest_city)
        unvisited_cities.remove(closest_city)
        current_city = closest_city

    return tour

coordinates = load_coordinates('tsp.txt')

distance_matrix = create_distance_matrix(coordinates)

greedy_solution = greedy_algorithm(distance_matrix)
greedy_fitness = fitness(greedy_solution, distance_matrix)
print(f"Greedy Solution: {greedy_solution} with fitness {greedy_fitness}")

num_cities = len(coordinates)
population_size = 1000
generations = 50
tournament_size = 5
crossover_rate = 0.8
mutation_rate = 0.01

start_GA = time.time()
# ga_solution = genetic_algorithm(num_cities, population_size, generations, tournament_size, crossover_rate, mutation_rate, distance_matrix)
ga_solution, best_fitness_over_generations, average_fitness_over_generations = genetic_algorithm(
    num_cities, population_size, generations, tournament_size, crossover_rate, mutation_rate, distance_matrix)
end_GA = time.time()

ga_fitness = fitness(ga_solution, distance_matrix)
print(f"Genetic Algorithm Solution: {ga_solution} with fitness {ga_fitness}")
print("Time taken for Genetic Algorithm: ", end_GA - start_GA)

with open("tsp_solution.txt", 'w') as file:
    for number in ga_solution:
        file.write(f"{number}\n")


plt.figure(figsize=(10, 6))
plt.plot(best_fitness_over_generations, label='Best Fitness')
plt.plot(average_fitness_over_generations, label='Average Fitness')
plt.title('Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.grid(True)
plt.show()