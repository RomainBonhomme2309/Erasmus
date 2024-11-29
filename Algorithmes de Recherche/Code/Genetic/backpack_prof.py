import random
import array
import numpy as np


class Configuration():
    RANDOMSEED = 12345
    POPULATIONSIZE = 50
    NOBJECTS = 40 # How many objects in total
    MAXWEIGHTS= 60
    MUTATIONRATE = 5 # 5% de mutation rate
    WEIGHTS = array.array('f')
    COSTS = array.array('f')

    def __init__(self):
        random.seed(self.RANDOMSEED)
        for i in range(0, self.NOBJECTS):
            self.WEIGHTS.append(random.uniform(1, 6))
            self.COSTS.append(random.uniform(20, 300))

config = Configuration()

class Chromosome:
    _value = None

    def __init__(self, initialValue=None):
        if initialValue is not None:
            self._value = initialValue
        else:
            self._initRandomOne()

    def fitness(self):
        pass

    def reproduceWith(self, other):
        pass

    def mutation(self):
        pass

    def _initRandomOne(self):
        pass

    def __repr__(self):
        return str(self._value)


class ChromosomeKS(Chromosome):
    _weight = None
    _cost = None

    def __init__(self, initialValue=None):
        self._weight = 0
        self._cost = 0
        #self._value = initialValue if initialValue is not None else self._randomOne()
        super().__init__(initialValue)

    def fitness(self):
        global config 
        return 0 if self._weight > config.MAXWEIGHTS else self._cost

    def reproduceWith(self, other):
        global config 
        # generates the two (empty) children
        toret = []
        for i in range(0,2):
            toret.append(ChromosomeKS(array.array('b',[0]*config.NOBJECTS)))

        crossover = random.randint(1,config.NOBJECTS-1)

        for i in range(0,crossover):
            toret[0]._value[i] = self._value[i]
            toret[1]._value[i] = other._value[i]
        for i in range(crossover, config.NOBJECTS):
            toret[0]._value[i] = other._value[i]
            toret[1]._value[i] = self._value[i]

        toret[0]._updateScores()
        toret[1]._updateScores()
        return toret

    def mutation(self):
        global config
        i = random.randint(0,len(self._value)-1)
        if self._value[i]==1:
            self._value[i] = 0
            self._weight -= config.WEIGHTS[i]
            self._cost -= config.COSTS[i]
        else: # We can mutate to a non valid solution
            self._value[i] = 1
            self._weight += config.WEIGHTS[i]
            self._cost += config.COSTS[i]

    def _updateScores(self):
        self._cost = sum(config.COSTS[i] for i in range(0,config.NOBJECTS) if self._value[i] == 1)
        self._weight = sum(config.WEIGHTS[i] for i in range(0,config.NOBJECTS) if self._value[i] == 1)

    def _initRandomOne(self):
        global config
        self._value = array.array('b',[0]*config.NOBJECTS)
        permutation = [x for x in range(0,config.NOBJECTS)] # permutation is needed to fill the chromosome
        random.shuffle(permutation)
        for i in range(0,config.NOBJECTS):
            ii = permutation[i]
            self._value[ii] = random.randint(0,1)
            if self._value[ii] == 1:
                if self._weight + config.WEIGHTS[ii] < config.MAXWEIGHTS: # Allow to greedy generate a viable solution
                    self._weight += config.WEIGHTS[ii]
                    self._cost += config.COSTS[ii]
                else:
                    self._value[ii] = 0 # Remove the object

    def __repr__(self):
        toret = ""
        for v in self._value:
            toret += "-" if v == 0 else "X"
        toret += " (W="+str(int(self._weight))+") (C=" + str(int(self._cost))+")"
        return toret 

class Population():
    _population = None
    _populationSize = None

    def __init__(self, populationsize = None):
        global config
        self._populationSize = populationsize if populationsize is not None else config.POPULATIONSIZE
        self.randomInit()

    def randomInit(self):
        self._population = []
        while len(self._population) < self._populationSize:
            self._population.append(ChromosomeKS())
        self._sort()

    def oneGeneration(self):
        ''' We assume a sorted list here'''
        global config
        newpop = self._population[:4]
        while len(newpop) < len(self):
            i1 = self.randomSelect()
            i2 = self.randomSelect(i1)
            for newson in i1.reproduceWith(i2):
                if random.randint(0,100) < config.MUTATIONRATE:
                    newson.mutation()
                newpop.append(newson)
        self._population = newpop
        self._populationSize = len(newpop)
        self._sort()

    def randomSelect(self, taboo=None):
        ''' We assume a sorted list here'''
        somme = (len(self._population)*(len(self._population)-1)) / 2 #sum(x.fitness() for x in self._population)
        current = len(self._population)
        cumul = current
        limit = random.randint(0,somme)
        i = 0
        while  cumul < limit:
            current -= 1
            cumul += current
            assert current > 0
            i += 1
        assert i < len(self._population)
        return self._population[i]
        
        return self._population[random.randint(0,len(self._population)-1)]

    def _sort(self):
        self._population.sort(key=lambda x: x.fitness(), reverse=True)

    def __repr__(self):
        toret = ""
        for i in self._population:
            toret += i.__repr__() + " (f="+ str(int(i.fitness()))+")\n"
        return toret

    def __len__(self):
        return len(self._population)

best = None
bestparameters = None
for k in range(0,20):
    config.RANDOMSEED = random.randint(10000,200000)
    config.POPULATIONSIZE = 20 + random.randint(1,10)**3
    config.MUTATIONRATE = 1 + random.randint(1,5)**2
    population = Population()
    # FILL HERE
    
print("Meilleur solution")
print(best)
print(bestparameters)
