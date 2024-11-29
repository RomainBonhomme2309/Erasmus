import random
import sys
import math
from individual import Individual  # Supposons que vous avez déjà défini "Individual" dans un autre fichier
from population import Population    # Supposons que vous avez déjà défini "Population" dans un autre fichier
from gifenpy import GIF               # Supposons que vous avez déjà défini "GIF" pour la création de GIFs


class Map:
    def __init__(self, nbTowns):
        self.nbTowns = nbTowns
        self.x = [0.0] * nbTowns
        self.y = [0.0] * nbTowns


def terminate(message):
    print(message, file=sys.stderr)
    exit(-1)


def get_field(input_string):
    return input_string.strip().split(',')


def tsp_load_map_from_file(filename, nbTowns):
    try:
        with open(filename, 'r') as fp:
            map = Map(nbTowns)

            for i in range(nbTowns):
                line = fp.readline()
                fields = get_field(line)
                map.x[i] = float(fields[1])
                map.y[i] = float(fields[2])

            print(f"Loaded {nbTowns} towns from {filename}")
            return map
    except Exception as e:
        terminate(f"loadMapFromFile: {str(e)}")


def tsp_free_map(map):
    del map.x
    del map.y
    del map


def tsp_tour_to_gif(tour, map, filename, size):
    minX = min(map.x)
    maxX = max(map.x)
    minY = min(map.y)
    maxY = max(map.y)

    xRange = maxX - minX
    yRange = maxY - minY

    if xRange > yRange:
        sizex = size
        sizey = int(size * yRange / xRange)
    else:
        sizex = int(size * xRange / yRange)
        sizey = size

    gif = GIF(filename, sizex + 10, sizey + 10, palette=[(255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0)], depth=2)

    prevX = (map.x[tour[-1]] - minX) * sizex / xRange + 5
    prevY = (map.y[tour[-1]] - minY) * sizey / yRange + 5

    # Clean frame
    gif.clear_frame()

    for pos in range(map.nbTowns):
        nextX = (map.x[tour[pos]] - minX) * sizex / xRange + 5
        nextY = (map.y[tour[pos]] - minY) * sizey / yRange + 5
        gif.draw_line(int(prevX), int(prevY), int(nextX), int(nextY), 2)
        prevX = nextX
        prevY = nextY

    gif.add_frame()
    gif.close()


def tsp_get_tour_length(tour, map):
    tour_length = 0.0

    for i in range(map.nbTowns - 1):
        dx = map.x[tour[i]] - map.x[tour[i + 1]]
        dy = map.y[tour[i]] - map.y[tour[i + 1]]
        tour_length += math.sqrt(dx * dx + dy * dy)

    # Ajouter le retour à la première ville
    dx = map.x[tour[0]] - map.x[tour[-1]]
    dy = map.y[tour[0]] - map.y[tour[-1]]
    tour_length += math.sqrt(dx * dx + dy * dy)

    return tour_length


def fitness(ind, param):
    map = param
    tour_villes = [ind.get_gene(i) for i in range(map.nbTowns)]
    tour_length = tsp_get_tour_length(tour_villes, map)
    return 1.0 / tour_length


def tsp_optimize_by_ga(map, nb_iterations, size_population, elite_size, pmutation, verbose):
    pop = Population(length=map.nbTowns, nbVal=map.nbTowns, size=size_population,
                     fitness_func=fitness, paramFitness=map,
                     init_func=Individual.random_perm_init,
                     mutation_func=Individual.perm_mutation,
                     pmutation=pmutation,
                     crossover_func=Individual.perm_crossover,
                     eliteSize=elite_size)

    for t in range(nb_iterations):
        pop.evolve()
        if verbose == 1:
            print(f"Step {t + 1}: Max fitness = {pop.get_max_fitness()}")

    best_ind = pop.get_best_individual()
    tour_villes = [best_ind.get_gene(i) for i in range(map.nbTowns)]

    with open("tour.txt", "w") as fp:
        best_ind.print_individual(fp)

    return tour_villes
