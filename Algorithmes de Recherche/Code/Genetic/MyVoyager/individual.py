import random

class Individual:
    def __init__(self, length, nbVal):
        self.length = length
        self.nbVal = nbVal
        self.genotype = [0] * length

    def get_length(self):
        return self.length

    def get_gene(self, i):
        return self.genotype[i]

    def set_gene(self, i, val):
        self.genotype[i] = val

    def random_init(self):
        self.genotype = [random.randint(0, self.nbVal - 1) for _ in range(self.length)]

    def random_perm_init(self):
        self.genotype = list(range(self.length))
        for i in range(self.length - 1, 0, -1):
            j = random.randint(0, i)
            self.swap(i, j)

    def copy(self):
        new_ind = Individual(self.length, self.nbVal)
        new_ind.genotype = self.genotype[:]
        return new_ind

    def print_individual(self, file):
        """Print the individual's genotype to a file."""
        print(", ".join(map(str, self.genotype)), file=file)


    def seq_mutation(self, pm):
        for i in range(self.length):
            if random.random() <= pm:
                self.genotype[i] = random.randint(0, self.nbVal - 1)

    def seq_crossover(self, parent2):
        p = random.randint(0, self.length)
        if p == 0:
            return parent2.copy()
        elif p == self.length:
            return self.copy()
        else:
            child = Individual(self.length, self.nbVal)
            for i in range(self.length):
                if i < p:
                    child.genotype[i] = self.genotype[i]
                else:
                    child.genotype[i] = parent2.genotype[i]
            return child

    def perm_mutation(self, pm):
        p_1, p_2 = random.sample(range(self.length), 2)
        if p_1 > p_2:
            p_1, p_2 = p_2, p_1
        if random.random() <= pm:
            while p_1 < p_2:
                self.swap(p_1, p_2)
                p_1 += 1
                p_2 -= 1

    def perm_crossover(self, parent2):
        child = self.copy()
        p_1, p_2 = sorted(random.sample(range(self.length), 2))
        position = {val: idx for idx, val in enumerate(self.genotype)}
        for j in range(p_1, p_2):
            temp1 = child.genotype[j]
            temp2 = parent2.genotype[j]
            child.swap(j, position[temp2])
            position[temp1], position[temp2] = position[temp2], position[temp1]
        return child

    def swap(self, i, j):
        self.genotype[i], self.genotype[j] = self.genotype[j], self.genotype[i]