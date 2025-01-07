from pacman_module.game import Agent
from pacman_module.game import Directions
from pacman_module.util import manhattanDistance
from collections import Counter


def player(depth):
    """Returns which player's turn it is.

    Arguments:
        depth: the depth of the current state.

    Returns:
        0 for the Pac-Man's turn and 1 for Ghost's turn.
    """

    # Ghost's turn = odd number
    if (depth % 2) != 0:
        return 1

    # Pac-Man's turn = even number
    else:
        return 0


def key(state):
    """Returns a key that identifies a Pacman game state.
    A state is defined by:
    The food matrix, the position of Pac-Man and the Ghosts,
    and the Ghosts direction (as it can't make a half-turn
    unless it has to, its direction has an impact on the game)

    Arguments:
        state: the current game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key object that identifies a Pacman game state.
    """

    return (state.getFood(), state.getPacmanPosition(),
            state.getGhostPosition(1), state.getGhostDirection(1))


class PacmanAgent(Agent):
    def __init__(self):
        super().__init__()

        # This list will contains all the moves Pacman will execute
        self.moves = []
        # Global variables
        global closedStates
        global foodDots
        closedStates = []
        foodDots = []

    def cutoff(self, state, depth):
        """Returns whether or not expansion of nodes should
        continue at the given node.

        Arguments:
            state: the current game state. See API or class `pacman.GameState`.
            depth: depth in the recursion tree.

        Returns:
            A boolean value to continue or not expanding nodes.
        """

        # Initialization of variables
        pPos = state.getPacmanPosition()
        gPos = state.getGhostPosition(1)

        # Recursion
        # 1-status of the state
        if state.isWin() is True or state.isLose() is True:
            return True

        # 2-status of the cell: if the
        # cell contains a food dot and the ghost is far
        # enough not to eat Pacman, it stops.

        # Distance between Pacman and the ghost
        ghostDist = manhattanDistance(pPos, gPos)
        if pPos in foodDots and ghostDist > 2:
            return True

        # 3- quiescent state
        elif ghostDist > (6 - depth) and ghostDist > 1:
            return True

        # 4-the depth
        elif depth == 5:
            return True

        else:
            return False

    def eval_state(self, state, depth):
        """Returns an evaluation of utility value of the state.

        Arguments:
            state: the current game state. See API or class `pacman.GameState`.
            depth: the level of the node containing 'state' in the tree.
        Returns:
            The evaluation of utility value at a given state.
        """

        # Initialization of variables
        pPos = state.getPacmanPosition()
        gPos = state.getGhostPosition(1)

        # Stores the different distances computed
        distancesFood = []

        # Foods' distance from Pacman
        for i in range(len(foodDots)):
            foodPos = foodDots[i]
            distancesFood.append(manhattanDistance(pPos, foodPos))

        # Ghost's distance from Pacman
        distanceGhost = manhattanDistance(pPos, gPos)

        # Computation of the returned value
        if len(distancesFood) == 0:
            return (state.getScore() + distanceGhost)
        else:
            return (state.getScore() - 3 * min(distancesFood) + distanceGhost)

    def hminimaxFct(self, node, closed):
        """Returns the utility score of a state.

        Arguments:
            node: tuple = state [0] and its depth in the
                tree [1]
            closed: list = all the states already explored

        Returns:
            The utility score of a given state.
        """

        # Newclosed contains all the visited states on the direct
        # path to the given state
        newclosed = closed.copy()

        # Adds the state's node to the list of state already explored
        newclosed.append(key(node[0]))

        # Verifies if the state is an ending state
        if self.cutoff(node[0], node[1]) is True:
            return self.eval_state(node[0], node[1])

        # The current state is not an ending state, the
        # algorithm goes deeper in the tree
        else:
            # Contains the values used to find the next best move
            values = []

            # Pacman's turn
            if player(node[1]) == 0:
                # Next best move in the successors
                for next_state, action in node[0].generatePacmanSuccessors():
                    newKey = key(next_state)
                    # Only look at states not explored
                    if newKey not in newclosed and newKey not in closedStates:
                        # Determine the utility value
                        newDepth = node[1] + 1
                        eval = self.hminimaxFct([next_state, newDepth],
                                                newclosed)

                        # Adds the utility value to the list
                        values.append(eval)
                return max(values)

            # Ghost's turn
            elif player(node[1]) == 1:
                # Next best move in the successors
                for next_state, action in node[0].generateGhostSuccessors(1):
                    newKey = key(next_state)

                    # Only look at states not explored
                    if newKey not in newclosed and newKey not in closedStates:
                        # Determine the utility value
                        newDepth = node[1] + 1
                        eval = self.hminimaxFct([next_state, newDepth],
                                                newclosed)

                        # Adds the utility value to the list
                        values.append(eval)
                return min(values)

    def hminimax(self, state):
        """Returns the optimal next move for pacman

        Arguments:
            state: the current game state. See FAQ and class
                   `pacman.GameState`.

        Returns:
            A move as defined in `game.Directions`.
        """

        # Initialization of variables
        # Contains all the states Pacman went through
        # between each call of h-minimax
        global closedStates

        # List of the position of every dot in the game
        global foodDots

        if len(foodDots) == 0:
            foodPosition = state.getFood()
            length = 0
            for boolean in foodPosition:
                length = length + 1
            width = len(foodPosition[0])
            i = 0
            while i < length:
                j = 0
                while j < width:
                    if foodPosition[i][j] is True:
                        foodDots.append((i, j))
                    j = j + 1
                i = i + 1

        # Contains the values used to find the next best move
        value = []

        # Contains a list of the state already explored for each successor
        closed = []

        # Contains the number of occurences of the successors
        occurence = []

        # Contains the next states and their action
        action_list = []
        action_list_next = []
        state_list = []
        state_list_next = []

        # If Pacman eats a dot, we have to remove the dot from the list
        if state.getPacmanPosition() in foodDots:
            foodDots.pop(foodDots.index(state.getPacmanPosition()))

        # Add the current state to the two list of visited ones
        closedStates.append(key(state))
        closed.append(key(state))

        # Looks for the next best move
        for nextState, action in state.generatePacmanSuccessors():
            newKey = key(nextState)
            action_list_next.append(action)
            state_list_next.append(nextState)

            # States not explored
            if newKey not in closed and newKey not in closedStates:
                # Stores a state and its depth in the tree
                node = [nextState, 1]

                # Utility value of the next state
                eval = self.hminimaxFct(node, closed)

                # Adds the utility value and the action of
                # the next state into the list
                value.append(eval)
                action_list.append(action)
                state_list.append(nextState)

        # Returns the best action
        if len(value) != 0:
            # Adds the state to the list of explored one
            index_best_action = value.index(max(value))
            closedStates.append(key(state_list[index_best_action]))
            return [action_list[index_best_action]]

        # All children have already been visited, we
        # look for an action in the less visited state
        elif len(value) == 0:
            # Count how many times each state in closedStates has been visited
            all_occurence = Counter(closedStates)

            # As all states has been visited at least once, the least visited
            # state is chosen as next state.
            b = 0
            for action_next in action_list_next:
                # Counts how many times the given state
                # appears in closed_states
                occurence.append(all_occurence[key(state_list_next[b])])
                b = b + 1

            # Removes losing state from the list
            c = 0
            for action_bis in action_list_next:
                if state_list_next[c].isLose() is True:
                    occurence.pop(c)
                    state_list_next.pop(c)
                    action_list_next.pop(c)
                c = c + 1

            # Retrieves the action related to the least visited state
            index_min = occurence.index(min(occurence))
            closedStates.append(key(state_list_next[index_min]))

            return [action_list_next[index_min]]

    def get_action(self, state):
        """Returns a legal move from a pacman game state.

        Arguments:
            state: the current game state. See FAQ and class
                   `pacman.GameState`.
        """

        if not self.moves:
            self.moves = self.hminimax(state)
        try:
            return self.moves.pop(0)

        # If there aren't avalaible actions
        except IndexError:
            return Directions.STOP
