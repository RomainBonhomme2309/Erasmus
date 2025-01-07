from pacman_module.game import Agent
from pacman_module.game import Directions
from pacman_module.util import PriorityQueue, manhattanDistance


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """

    return (state.getPacmanPosition(),
            state.getFood(),
            tuple(state.getCapsules()))


def heuristic(state):
    """Define the heuristic function of the best path. In this case,
    the function will compute the distance to the furthest food.

    Arguments:
        state : a game state. See API or class `pacman.GameState`.

    Returns:
        The maximum value found.
    """

    pacman_position = state.getPacmanPosition()
    grid = state.getFood()

    max_distance = 0

    for i in range(grid.width):
        for j in range(grid.height):
            if grid[i][j]:
                actual_distance = manhattanDistance((i, j), pacman_position)
                if actual_distance > max_distance:
                    max_distance = actual_distance

    return max_distance


def cost_move(current_state, next_state):
    """Define the heuristic function of the best path. In this case,
    the function will be ...

    Arguments:
        current_state : the actual game state.
        next_state : the state that will be reached after the move.

    Returns:
        The cost of the move.
    """
    current_capsules = current_state.getCapsules()
    next_capsules = next_state.getCapsules()

    # A capsule has been eaten
    if len(next_capsules) < len(current_capsules):
        return 6

    # No capsule has been eaten
    else:
        return 1


class PacmanAgent(Agent):
    """Empty Pacman agent."""

    def __init__(self):
        super().__init__()

        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        if not self.moves:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def astar(self, state):
        """Given a Pacman game state, returns a list of legal moves to solve
        the search layout using the A* algorithm.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """

        path = []
        fringe = PriorityQueue()
        fringe.push((state, path, 0), 0)
        closed = set()

        while True:
            if fringe.isEmpty():
                return []  # failure during the instanciation

            _, (current, path, cost) = fringe.pop()

            if current.isWin():
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            for next_state, action in current.generatePacmanSuccessors():
                next_cost = cost + cost_move(current, next_state)
                next_heuristic = next_cost + heuristic(next_state)
                fringe.push((next_state, path + [action], next_cost),
                            next_heuristic)

        return path
