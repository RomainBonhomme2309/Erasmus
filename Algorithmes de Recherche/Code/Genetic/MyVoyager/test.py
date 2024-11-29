import sys
import random
from individual import Individual  # Assurez-vous que vous avez défini cette classe
from population import Population    # Assurez-vous que vous avez défini cette classe
from tsp import tsp_load_map_from_file, tsp_optimize_by_ga, tsp_get_tour_length, tsp_tour_to_gif  # Importez vos fonctions TSP ici

# Retourne le nombre de 1 dans une séquence binaire
def num1_in_seq(ind):
    return sum(1.0 for i in range(ind.get_length()) if ind.get_gene(i) > 0)

# Retourne la valeur de la fonction f(x) = -x^2/10 + 3x + 4 pour un x donné
def f(ind):
    x = 0
    for i in range(ind.get_length()):
        x = 2 * x + ind.get_gene(i)
    return -x * x / 10.0 + 3.0 * x + 4

# Démarre l'algorithme génétique pour maximiser le nombre de 1
def maximize_number_of_1():
    pop = Population(length=20, nb_val=2, size=100,
                     fitness_func=num1_in_seq, paramFitness=None,
                     init_func=Individual.random_init,
                     mutation_func=Individual.seq_mutation, 
                     pmutation=0.01,
                     crossover_func=Individual.seq_crossover, 
                     eliteSize=2)

    for t in range(100):
        pop.evolve()
        print(f"Step {t + 1:3}: Max fitness = {pop.get_max_fitness()}")

    print("Best individual found: ", end="")
    best_ind = pop.get_best_individual()
    best_ind.print_individual()
    print(f", with fitness of: {pop.get_max_fitness()}")

    pop.free()

# Démarre l'algorithme génétique pour maximiser la fonction
def maximize_function():
    pop = Population(length=5, nb_val=2, size=6,
                     fitness_func=f, param=None,
                     init_func=Individual.random_init,
                     mutation_func=Individual.seq_mutation, 
                     pmutation=0.1,
                     crossover_func=Individual.seq_crossover, 
                     elite_size=2)

    for t in range(30):
        pop.evolve()
        print(f"Step {t + 1:3}: Max fitness = {pop.get_max_fitness()}")

    print("Best individual found: ", end="")
    best_ind = pop.get_best_individual()
    best_ind.print_individual()
    print(f", with fitness of: {pop.get_max_fitness()}")

    pop.free()

# Démarre l'algorithme génétique pour le problème du voyageur de commerce
def tsp(argc, argv):
    if argc != 8:
        print("Usage: python test_ga.py 3 <file> <ntowns> <size> <pm> <elitesize> <niterations>", file=sys.stderr)
        exit(-1)

    filename = argv[2]
    n_towns = int(argv[3])
    size = int(argv[4])
    pm = float(argv[5])
    elite_size = int(argv[6])
    nb_iterations = int(argv[7])

    map = tsp_load_map_from_file(filename, n_towns)

    min_length = float('inf')
    best_tour = tsp_optimize_by_ga(map, nb_iterations, size, elite_size, pm, verbose=0)
    min_length = tsp_get_tour_length(best_tour, map)

    meta_iters = 30
    for i in range(meta_iters):
        print(f"{i} ", end="")
        temp_tour = tsp_optimize_by_ga(map, nb_iterations, size, elite_size, pm, verbose=0)
        length = tsp_get_tour_length(temp_tour, map)

        if length < min_length:
            best_tour = temp_tour
            min_length = length
        else:
            # Supposons que vous avez une méthode pour libérer la mémoire en Python si nécessaire
            del temp_tour

    print("\n")
    print(f"{filename} {n_towns} {size} {pm:.3f} {elite_size} {nb_iterations}")
    print(f"Best tour found: (length = {min_length})", end="")
    
    for town in best_tour:
        print(f" {town}", end="")
    print()

    tsp_tour_to_gif(best_tour, map, "tour.gif", size=1000)

# Point d'entrée principal
if __name__ == "__main__":
    random.seed()  # Initialise le générateur de nombres aléatoires

    if len(sys.argv) == 1:
        print("Usage: python test_ga.py <mode>.", file=sys.stderr)
        exit(-1)

    mode = int(sys.argv[1])
    if mode == 1:
        maximize_number_of_1()
    elif mode == 2:
        maximize_function()
    elif mode == 3:
        tsp(len(sys.argv), sys.argv)
    else:
        print(f"<mode> should be 1, 2 or 3, got {mode}", file=sys.stderr)
        exit(-1)


# France : Montpelier - Lyon - Clermont-Ferrand - Bordeaux - Poitiers - Angers -
#          Nantes - Brest - Caen - Paris - Lille - Strasbourg - Nice