from random import randint, getrandbits, choice
import numpy
from collections import namedtuple

__boardSize__ = 4

class Board:
    _data = None

    def __init__(self, copyFrom=None, randomize = False):
        if copyFrom is None:
            fromList = [x for x in range(0,__boardSize__**2)]
            if randomize:
                fromList = sorted(fromList, key = lambda x: getrandbits(1))
            self._data = numpy.reshape(fromList, (__boardSize__, __boardSize__))
        else:
            self._data = numpy.array(copyFrom, copy=True)

    def empty(self):
        p = numpy.where(self._data==0)
        return (p[0][0], p[1][0])

    def _inBound(self, x):
        return x >= 0 and x < __boardSize__

    def randomMove(self):
        p = self.empty()
        assert self._data[p[0], p[1]] == 0
        move = choice([(-1,0), (+1,0), (0,-1), (0,+1)])
        if self._inBound(move[0]+p[0]) and self._inBound(move[1]+p[1]): 
            self._data[p[0],p[1]] = self._data[p[0]+move[0], p[1]+move[1]]
            self._data[p[0]+move[0], p[1]+move[1]] = 0

    def next(self):
        """
        Generates the next possible states of the board by moving the empty tile in all four directions.

        Returns:
            list: A list of Board objects representing the next possible states.
        """
        toret= []
        e = self.empty()
        for x,y in ((-1,0), (+1,0), (0,-1), (0,+1)):
            if self._inBound(x+e[0]) and self._inBound(y+e[1]):
               n = Board(copyFrom=self._data)
               n._data[e[0],e[1]] = n._data[x+e[0], y+e[1]]
               n._data[x+e[0], y+e[1]] = 0
               toret.append(n)
        return toret

    def __eq__(self, other):
        return numpy.array_equal(self._data, other._data)

    def __hash__(self):
        self._data.flags.writeable = False
        h = hash(self._data.data.tobytes())
        self._data.flags.writeable = True
        return h 


class Node:
    state = None
    father = None
    g = None
    f = None 

    def __init__(self, state, father = None, g = 0, f = 0):
        self.state = state
        self.father = father
        self.g = g
        self.f = f

    # Permet de savoir si deux noeuds sont identiques
    def sameAsState(self, state):
        return (state is self.state) or (numpy.array_equal(self.state._data, state._data))

class Frontiere:
    _nodes = None 

    def __init__(self):
       self._nodes = [] 

    # Fonction qui permet de récupérer le noeud suivant dans la frontiere
    def getNext(self):
        return self._nodes.pop(0)

    # Fonction qui récupère un noeud d'apres son etat. Vous devez la rendre plus efficace
    def getNodeByState(self, state):

        for n in self._nodes:
            if n.sameAsState(state):
                return n
        return None

    # Fonction permettant d'ajouter un noeud à la frontiere 
    def addNode(self, state, father = None, g = 0, checkAlreadyThere = False):
        self._nodes.append(state)
        self._nodes.sort(key=lambda n: n.f)  # Sorting by the cost f 

    def __len__(self):
        return len(self._nodes)

    def size(self):
        return len(self._nodes)


def manhattan_distance(board, goal):
    """
    Calculate the Manhattan distance heuristic for A*.
    """
    distance = 0
    for x in range(__boardSize__):
        for y in range(__boardSize__):
            value = board._data[x, y]
            if value != 0:  # Ignore the empty space (represented by 0)
                goal_x, goal_y = divmod(value, __boardSize__)
                distance += abs(goal_x - x) + abs(goal_y - y)
    return distance

def uniform_cost_search(initial_state, goal_state):
    """
    Uniform Cost Search (UCS) implementation.
    This uses g(n), the cost from the start node to the current node, without any heuristics.
    """
    frontiere = Frontiere()
    closed = set()
    initial_node = Node(initial_state, g=0, f=0)  # For UCS, f = g
    frontiere.addNode(initial_node)
    
    cost_so_far = {hash(initial_state): 0}

    while frontiere.size() > 0:
        current_node = frontiere.getNext()

        if current_node.state.__eq__(goal_state):
            return reconstruct_path(current_node)

        closed.add(current_node.state)

        for next_state in current_node.state.next():
            new_g = current_node.g + 1  # Increment cost by 1 for each move
            next_state_hash = hash(next_state)

            if next_state in closed:
                continue

            if next_state_hash not in cost_so_far or new_g < cost_so_far[next_state_hash]:
                cost_so_far[next_state_hash] = new_g
                new_node = Node(next_state, father=current_node, g=new_g, f=new_g)  # For UCS, f = g
                frontiere.addNode(new_node)

    return None  # No solution found

def a_star_search(initial_state, goal_state):
    """
    A* search algorithm implementation.
    This uses g(n) + h(n), where g(n) is the cost from the start node and h(n) is the Manhattan distance heuristic.
    """
    frontiere = Frontiere()
    closed = set()
    initial_node = Node(initial_state, g=0, f=manhattan_distance(initial_state, goal_state))
    frontiere.addNode(initial_node)
    
    cost_so_far = {hash(initial_state): 0}

    while frontiere.size() > 0:
        current_node = frontiere.getNext()

        if current_node.state.__eq__(goal_state):
            return reconstruct_path(current_node)

        closed.add(current_node.state)

        for next_state in current_node.state.next():
            new_g = current_node.g + 1
            next_state_hash = hash(next_state)

            if next_state in closed:
                continue

            h = manhattan_distance(next_state, goal_state)
            new_f = new_g + h

            if next_state_hash not in cost_so_far or new_g < cost_so_far[next_state_hash]:
                cost_so_far[next_state_hash] = new_g
                new_node = Node(next_state, father=current_node, g=new_g, f=new_f)
                frontiere.addNode(new_node)

    return None

def reconstruct_path(node):
    """
    Reconstructs the path from the initial state to the goal state.
    """
    path = []
    while node is not None:
        path.append(node.state._data)
        node = node.father
    path.reverse()
    return path


# Code qui prrend la liberté de Python d'écrire en dehors de toute fonction

frontiere = Frontiere()
closed = set() # Dictionary of seen boards 

boardGoal = Board(randomize = False) # Génération du taquin "but"
# Exemple d'un taquin bien mélangé
# boardInit = Board(copyFrom=[[1, 4, 2],[5, 8, 7],[0, 3, 6]], randomize = False) 
# Exemple de construction d'un etat initial en mélangeant les pièces
boardInit = Board(copyFrom=boardGoal._data)
for i in range(0,50): # Plus vous mélangerez plus ce sera difficile
    boardInit.randomMove()

print("Initial Board:")
print(boardInit._data)

print("\nSolving with Uniform Cost Search:")
solution_ucs = uniform_cost_search(boardInit, boardGoal)
if solution_ucs:
    print(len(solution_ucs))
    for step in solution_ucs:
        print(step)
else:
    print("No solution found.")

print("\nSolving with A* Search:")
solution_a_star = a_star_search(boardInit, boardGoal)
if solution_a_star:
    print(len(solution_a_star))
    for step in solution_a_star:
        print(step)
else:
    print("No solution found.")


# Tableau avec : nb shuffle total cost nb iter #seen #seen+frontiere