import random
from individual import Individual  # Supposons que vous avez déjà défini "Individual" dans un autre fichier


class Ind:
    def __init__(self, individu=None, val_fitness=0.0):
        self.individu = individu
        self.val_fitness = val_fitness
        self.g = 0.0


class Population:
    def __init__(self, length, nbVal, size, fitness_func, paramFitness, init_func, mutation_func, pmutation, crossover_func, eliteSize):
        self.size = size
        self.length = length
        self.nbVal = nbVal
        self.fitness_func = fitness_func
        self.init_func = init_func
        self.mutation_func = mutation_func
        self.pmutation = pmutation
        self.crossover_func = crossover_func
        self.eliteSize = eliteSize
        self.paramFitness = paramFitness
        self.sumFitness = 0.0

        # Initialiser la population
        self.ind = [Ind() for _ in range(size)]

        for i in range(size):
            self.ind[i].individu = Individual(length, nbVal)
            self.init_func(self.ind[i].individu)
            self.ind[i].val_fitness = self.fitness_func(self.ind[i].individu, paramFitness)
            self.sumFitness += self.ind[i].val_fitness

        self.ind.sort(key=lambda x: x.val_fitness)

        sumTemp = 0
        for i in range(size):
            sumTemp += self.ind[i].val_fitness
            self.ind[i].g = sumTemp / self.sumFitness

    def get_max_fitness(self):
        return self.ind[-1].val_fitness

    def get_min_fitness(self):
        return self.ind[0].val_fitness

    def get_avg_fitness(self):
        return self.sumFitness / self.size

    def get_best_individual(self):
        return self.ind[-1].individu

    def selection(self):
        low = 0
        high = self.size - 1
        random_value = random.random()

        while low < high:
            medium = (low + high) // 2
            if random_value < self.ind[medium].g:
                high = medium
            else:
                low = medium + 1

        return self.ind[high].individu

    def evolve(self):
        eliteSize = self.eliteSize
        size = self.size
        new_population = [Ind() for _ in range(size)]

        # Conserver le meilleur individu
        new_population[size - 1].individu = self.ind[-1].individu.copy()
        new_population[size - 1].val_fitness = self.ind[-1].val_fitness
        self.sumFitness = new_population[size - 1].val_fitness

        # Mutation des individus d'élite
        for i in range(size - 2, size - eliteSize - 1, -1):
            new_population[i].individu = self.ind[i].individu.copy()
            self.mutation_func(new_population[i].individu, self.pmutation)
            new_population[i].val_fitness = self.fitness_func(new_population[i].individu, self.paramFitness)
            self.sumFitness += new_population[i].val_fitness

        # Croisement et mutation pour le reste de la population
        for i in range(size - eliteSize):
            parent1 = self.selection()
            parent2 = self.selection()
            new_population[i].individu = self.crossover_func(parent1, parent2)
            self.mutation_func(new_population[i].individu, self.pmutation)
            new_population[i].val_fitness = self.fitness_func(new_population[i].individu, self.paramFitness)
            self.sumFitness += new_population[i].val_fitness

        # Tri de la nouvelle population par valeur de fitness
        new_population.sort(key=lambda x: x.val_fitness)

        sumTemp = 0
        for i in range(size):
            sumTemp += new_population[i].val_fitness
            new_population[i].g = sumTemp / self.sumFitness

        # Libérer l'ancienne population
        self.ind = new_population

    def free(self):
        for ind in self.ind:
            ind.individu.free()
