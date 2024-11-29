import numpy as np

def backpack_iterative(values, weights, W):
    n = len(values)
    dp = [[0 for x in range(W + 1)] for x in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][W]

np.random.seed(42)
values = np.random.randint(0, 50, 100)
#values = values ** 2
weights = np.random.randint(0, 50, 100)
W = 1000

max_value = backpack_iterative(values, weights, W)
print(f"The maximum value that can be obtained with a classic method is {max_value}")


# Define the genetic algorithm-based backpack problem solver
class GeneticBackpack:
    def __init__(self, values, weights, capacity, population_size=500, generations=500, mutation_rate=0.05):
        self.values = values
        self.weights = weights
        self.capacity = capacity
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.n_items = len(values)

    # Initialize population with random binary chromosomes
    def initialize_population(self):
        return np.random.randint(2, size=(self.population_size, self.n_items))

    # Fitness function to evaluate total value while respecting the weight constraint
    def fitness(self, chromosome):
        total_value = np.sum(chromosome * self.values)
        total_weight = np.sum(chromosome * self.weights)
        if total_weight > self.capacity:
            return 0  # Penalize invalid solutions
        return total_value

    # Roulette wheel selection for choosing parents based on fitness
    def selection(self, population, fitness_values):
        total_fitness = np.sum(fitness_values)
        if total_fitness == 0:
            return population[np.random.randint(0, self.population_size)]
        
        probabilities = fitness_values / total_fitness
        return population[np.random.choice(range(self.population_size), p=probabilities)]

    # Crossover: Single-point crossover
    def crossover(self, parent1, parent2):
        point = np.random.randint(1, self.n_items - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2

    # Mutation: Randomly flip bits in the chromosome
    def mutate(self, chromosome):
        for i in range(self.n_items):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    # The main genetic algorithm loop
    def run(self):
        population = self.initialize_population()
        
        for generation in range(self.generations):
            fitness_values = np.array([self.fitness(ind) for ind in population])

            # Create new population
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.selection(population, fitness_values)
                parent2 = self.selection(population, fitness_values)

                # Apply crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Apply mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)
                new_population.append(child2)

            population = np.array(new_population[:self.population_size])

            # Output best solution in current generation
            best_fitness = np.max(fitness_values)
            best_individual = population[np.argmax(fitness_values)]
            # print(f"Generation {generation + 1}: Best fitness = {best_fitness}")

        # Return best solution found
        best_fitness = np.max(fitness_values)
        best_individual = population[np.argmax(fitness_values)]
        return best_individual, best_fitness

# Example usage
np.random.seed(42)
values = np.random.randint(0, 50, 100)
# values = values ** 2
weights = np.random.randint(0, 50, 100)
W = 1000

ga = GeneticBackpack(values, weights, W)
best_solution, best_value = ga.run()

print(values)
print(weights)

print(f"The best solution found is {best_solution}")
print(f"The maximum value obtained is {best_value}")
print(sum([weights[i] if best_solution[i] == 1 else 0 for i in range(len(best_solution))]))
