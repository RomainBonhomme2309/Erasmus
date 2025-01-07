from pacman_module.game import Agent
from pacman_module.game import Directions
import math


class PacmanAgent(Agent):
    def __init__(self):
        super().__init__()

        self.nb_ghosts = 0

    def get_info(self, state):
        """Returns basics information about the current game state

        Arguments:
            state: the current game state.

        Returns:
            A tuple containing the hash value of Pacman's position,
            a tuple containing the ghosts positions and the hash
            value of the food grid.
        """

        pos = state.getPacmanPosition()
        ghost_pos = state.getGhostPositions()
        food = state.getFood()

        return tuple([hash(pos), tuple(ghost_pos), hash(food)])

    def minimum_value(self, state, visited, ghost):
        """
        Arguments:
            state: the current game state.
            ghost: the position of the ghost agent.

        Returns:
            The minimum utility value of the successors.
        """

        if state.isLose() or state.isWin():
            return state.getScore()

        visited.add(self.get_info(state))
        successors = state.generateGhostSuccessors(ghost)
        min_value = math.inf

        for next in successors:
            if self.get_info(next[0]) not in visited:
                new_visited = visited.copy()

                if ghost > 1:
                    min_value = min(min_value,
                                    self.minimum_value(next[0],
                                                       new_visited,
                                                       ghost - 1))

                else:
                    min_value = min(min_value,
                                    self.maximum_value(next[0],
                                                       new_visited,
                                                       self.nb_ghosts))

        if min_value == math.inf:
            min_value = - min_value

        return min_value

    def maximum_value(self, state, visited, ghost):
        """
        Arguments:
            state: the current game state.
            ghost: the position of the ghost agent.

        Returns:
            The minimum utility value of the successors.
        """

        if state.isLose() or state.isWin():
            return state.getScore()

        visited.add(self.get_info(state))
        successors = state.generatePacmanSuccessors()
        max_value = - math.inf

        for succ in successors:
            if self.get_info(succ[0]) not in visited:
                new_visited = visited.copy()

                max_value = max(max_value, self.minimum_value(succ[0],
                                                              new_visited,
                                                              ghost))

        if max_value == -math.inf:
            max_value = - max_value

        return max_value

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move according to the
            Minimax algorithm.

        Arguments:
            state: the current game state. See API or class `pacman.GameState`.

        Returns:
            The best legal move, as defined in `game.Directions`,
            found by the Minimax algorithm.
        """

        visited = set()
        best_action = Directions.STOP
        best_value = - math.inf

        # Put the correct number of ghosts present in the grid
        self.nb_ghosts = state.getNumAgents() - 1

        visited.add(self.get_info(state))
        successors = state.generatePacmanSuccessors()

        for next_state, next_action in successors:
            value = self.minimum_value(next_state, visited, self.nb_ghosts)
            if value > best_value:
                best_value = value
                best_action = next_action

        return best_action
