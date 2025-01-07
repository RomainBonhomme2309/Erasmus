from pacman_module.game import Agent
from pacman_module.game import Directions
from pacman_module.util import manhattanDistance
import math


def key(state):
    """
    Returns a key that identifies a Pacman game state.
    A state is defined by:
    The food matrix, the position of Pac-Man and the Ghosts,
    and the Ghosts direction (as it can't make a half-turn
    unless it has to, its direction has an impact on the game)

    Arguments:
        state: the current game state. See API or class `pacman.GameState`.
    Return:
        A hashable key object that identifies a Pacman game state.
    """

    return (state.getPacmanPosition(), state.getGhostPosition(1),
            state.getGhostDirection(1), state.getFood())


class PacmanAgent(Agent):
    """Empty Pacman agent."""

    def __init__(self):
        super().__init__()

    def eval_state(self, state):
        """Returns an evaluation of utility value of the state.

        Arguments:
            state: the current game state. See API or class `pacman.GameState`.

        Returns:
            The evaluation of utility value at a given state.
        """

        foodList = state.getFood().asList()
        score = state.getScore()
        ghostList = state.getGhostPositions()
        pacmanPos = state.getPacmanPosition()

        # Final state
        if state.isWin() or state.isLose():
            return score

        # Distance from pacman to the closest ghost
        ghostDist = [manhattanDistance(pacmanPos, ghostPos) for ghostPos in
                     ghostList]
        closestGhost = min(ghostDist)
        scoreGhost = -(1/closestGhost)

        # Number of food left
        nbFood = state.getNumFood()

        # Distance from pacman to the closest food
        foodDist = [manhattanDistance(pacmanPos, foodPos) for foodPos in
                    foodList]
        closestFood = min(foodDist)
        scoreFood = -closestFood - nbFood

        # Score evaluation
        scoreFinal = score + scoreFood + 100 * scoreGhost

        return scoreFinal

    def cutoff(self, state, depth):
        """Returns whether or not expansion of nodes should continue
        at the given node.

        Arguments:
            state: the current game state. See API or class `pacman.GameState`.
            depth: depth in the recursion tree.

        Returns:
        A boolean value to continue or not expanding nodes.
        """

        return state.isWin() or state.isLose() or depth >= 11

    def hminimax(self, state, player, alpha, beta, depth, visited, computed):
        """Returns the minimax value of a node and the action to choose
        from that node. It gives the best action to execute as Pac-Man.

        Arguments:
            state: the current game state. See API or class `pacman.GameState`.
            player: 0 for Pac-Man and 1 for ghosts.
            alpha: Lower bound value for Alpha-Beta pruning.
            beta: Upper bound value for Alpha-Beta pruning.
            depth: Depth in the recursion tree.
            visited: A set containing the states already visited
            in the current traceback.

        Return:
            The minimax value of a node and the action to choose
            from that node.
        """

        # Check if cutoff is reached
        if self.cutoff(state, depth):
            return self.eval_state(state), None

        gameState = key(state)

        if player == 0:
            otherPlayer = 1
        else:
            otherPlayer = 0

        # Return states that have already been computed, if score is the same.
        if gameState in computed:
            return computed[gameState]

        # Initial minimax value
        if player == 0:
            minimaxValue = -math.inf
        else:
            minimaxValue = math.inf

        # Action towards the best path for Pac-Man.
        best_action = Directions.STOP
        successorMoves = state.generatePacmanSuccessors() if player == 0 else state.generateGhostSuccessors(player)

        for result, action in successorMoves:
            new_key = key(result)

            if new_key not in visited:
                visited.add(new_key)
                value = self.hminimax(result, otherPlayer, alpha, beta,
                                    depth + 1, visited, computed)[0]
                visited.remove(new_key)

                if player == 0:
                    # If player = Pac-Man, maximize the utility value.
                    if value > minimaxValue:
                        minimaxValue = value
                        best_action = action

                    # Pruning
                    if value >= beta:
                        if depth == 0:
                            computed[gameState] = value, action
                        return value, action

                    alpha = max(alpha, value)
                else:
                    # If player = Ghost, minimize the utility value.
                    if value < minimaxValue:
                        minimaxValue = value
                        best_action = action

                    # Pruning
                    if value <= alpha:
                        if depth == 0:
                            computed[gameState] = value, action
                        return value, action

                    beta = min(beta, value)

        # Add result to the transposition table.
        # Dictionary key = (state hash key, score)
        if depth == 0:
            computed[gameState] = minimaxValue, best_action

        return minimaxValue, best_action

    def get_action(self, state):
        visited = set()
        computed = dict()

        _, action = self.hminimax(state, 0, -math.inf, +math.inf,
                                0, visited, computed)
        print(action)
        return action
