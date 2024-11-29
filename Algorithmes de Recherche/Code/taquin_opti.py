from random import randint, getrandbits, choice
import numpy as np
from collections import namedtuple
import heapq

__boardSize__ = 3

class Board:
    def __init__(self, copyFrom=None, randomize=False):
        if copyFrom is None:
            fromList = [x for x in range(0, __boardSize__**2)]
            if randomize:
                fromList = sorted(fromList, key=lambda x: getrandbits(1))
            self._data = np.reshape(fromList, (__boardSize__, __boardSize__))
        else:
            self._data = np.array(copyFrom, copy=True)

    def empty(self):
        p = np.where(self._data == 0)
        return (p[0][0], p[1][0])

    def _inBound(self, x):
        return 0 <= x < __boardSize__

    def randomMove(self):
        p = self.empty()
        assert self._data[p[0], p[1]] == 0
        move = choice([(-1, 0), (+1, 0), (0, -1), (0, +1)])
        if self._inBound(move[0] + p[0]) and self._inBound(move[1] + p[1]):
            self._data[p[0], p[1]] = self._data[p[0] + move[0], p[1] + move[1]]
            self._data[p[0] + move[0], p[1] + move[1]] = 0

    def next(self):
        toret = []
        e = self.empty()
        for x, y in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
            if self._inBound(x + e[0]) and self._inBound(y + e[1]):
                n = Board(copyFrom=self._data)
                n._data[e[0], e[1]] = n._data[x + e[0], y + e[1]]
                n._data[x + e[0], y + e[1]] = 0
                toret.append(n)
        return toret

    def __eq__(self, other):
        return np.array_equal(self._data, other._data)

    def __hash__(self):
        self._data.flags.writeable = False
        h = hash(self._data.data.tobytes())
        self._data.flags.writeable = True
        return h

    def manhattan_distance(self, goal):
        """
        Calculate the Manhattan distance heuristic to the goal state.
        """
        distance = 0
        for x in range(__boardSize__):
            for y in range(__boardSize__):
                value = self._data[x, y]
                if value != 0:
                    goal_x, goal_y = divmod(value, __boardSize__)
                    distance += abs(goal_x - x) + abs(goal_y - y)
        return distance

    def is_goal(self, goal):
        return np.array_equal(self._data, goal._data)

class Node:
    def __init__(self, state, father=None, g=0, f=0):
        self.state = state
        self.father = father
        self.g = g
        self.f = f

    def __lt__(self, other):
        return self.f < other.f

class Frontiere:
    def __init__(self):
        self._nodes = []

    def getNext(self):
        return heapq.heappop(self._nodes)

    def addNode(self, node):
        heapq.heappush(self._nodes, node)

    def is_empty(self):
        return len(self._nodes) == 0

def reconstruct_path(node):
    path = []
    while node is not None:
        path.append(node.state._data)
        node = node.father
    path.reverse()
    return path

def uniform_cost_search(initial_state, goal_state):
    frontiere = Frontiere()
    closed = set()
    initial_node = Node(initial_state, g=0, f=0)
    frontiere.addNode(initial_node)

    while not frontiere.is_empty():
        current_node = frontiere.getNext()

        if current_node.state.is_goal(goal_state):
            return reconstruct_path(current_node)

        closed.add(current_node.state)

        for next_state in current_node.state.next():
            if next_state in closed:
                continue

            new_g = current_node.g + 1
            new_node = Node(next_state, father=current_node, g=new_g, f=new_g)

            frontiere.addNode(new_node)

    return None

def a_star_search(initial_state, goal_state):
    frontiere = Frontiere()
    closed = set()
    initial_node = Node(initial_state, g=0, f=0)
    initial_node.f = initial_node.g + initial_state.manhattan_distance(goal_state)
    frontiere.addNode(initial_node)

    while not frontiere.is_empty():
        current_node = frontiere.getNext()

        if current_node.state.is_goal(goal_state):
            return reconstruct_path(current_node)

        closed.add(current_node.state)

        for next_state in current_node.state.next():
            if next_state in closed:
                continue

            new_g = current_node.g + 1
            h = next_state.manhattan_distance(goal_state)
            new_node = Node(next_state, father=current_node, g=new_g, f=new_g + h)

            frontiere.addNode(new_node)

    return None

# Test the implementation
boardGoal = Board(randomize=False)  # Target (solved) state
boardInit = Board(copyFrom=boardGoal._data)

# Scramble the initial board
for i in range(10):  # Adjust the number of moves for more complexity
    boardInit.randomMove()

print("Initial board:")
print(boardInit._data)

print("\nSolving with Uniform Cost Search:")
solution_ucs = uniform_cost_search(boardInit, boardGoal)
if solution_ucs:
    for step in solution_ucs:
        print(step)
else:
    print("No solution found.")

print("\nSolving with A* Search:")
solution_a_star = a_star_search(boardInit, boardGoal)
if solution_a_star:
    for step in solution_a_star:
        print(step)
else:
    print("No solution found.")
