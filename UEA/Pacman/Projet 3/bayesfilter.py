import numpy as np
from pacman_module.game import Agent
from pacman_module.game import Directions
from pacman_module.game import manhattanDistance


def heuristic(position, pellets):
    """Define the heuristic function of the best path. In this case,
    the function will compute the distance between the different "pellets".
    These are the positions that satisfies the following condition :

        np.argwhere(beliefs[i] >= 0.99 * beliefs[i].max())

    Arguments:
        position : the position of a neighbour of pacman.
        pellets : positions that satisfies a condition on the belief.

    Returns:
        The sum of the maximum distance between two pellets and the minimum
        distance between pacman and these two pellets.
    """

    # Compute the distance between every pellets
    distances = np.array(
        [[manhattanDistance(pellets[i], pellets[j])
          for j in range(len(pellets))]
         for i in range(len(pellets))]
    )

    # Index of the maximum distance
    (i, j) = np.unravel_index(np.argmax(distances), distances.shape)

    max_distance = distances[i][j]
    min_distance = min((manhattanDistance(position, pellets[i])),
                       (manhattanDistance(position, pellets[j])))

    return (min_distance + max_distance)


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """Builds the transitiion matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (i, j) | X_{t-1} = (k, l)) for
            the ghost to move from (k, l) to (i, j).
        """

        width = walls.width
        height = walls.height
        trans_matrix = np.zeros((width, height, width, height))

        ghost_type = self.ghost

        # Probability of the movment doing by the ghost according to its type
        if (ghost_type == "afraid"):
            away = 2 ** 1
            closer = 1.0
        elif (ghost_type == "fearless"):
            away = 2 ** 0
            closer = 1.0
        elif (ghost_type == "terrified"):
            away = 2 ** 3
            closer = 1.0
        else:
            return None

        for w in range(width):
            for h in range(height):
                if (not (walls[w][h])):
                    neighbours = ((w - 1, h), (w + 1, h),
                                  (w, h - 1), (w, h + 1))

                    distance = manhattanDistance(position, (w, h))

                    for (i, j) in neighbours:
                        if (not (walls[i][j])):
                            neighbours_distance = manhattanDistance(position,
                                                                    (i, j))

                            if (neighbours_distance >= distance):
                                trans_matrix[i, j, w, h] = away
                            else:
                                trans_matrix[i, j, w, h] = closer

                    # Normalization of the matrix
                    trans_matrix[:, :, w, h] /= trans_matrix[:, :, w, h].sum()

        return trans_matrix

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        width = walls.width
        height = walls.height
        obser_matrix = np.zeros((width, height))

        for w in range(width):
            for h in range(height):
                distance = manhattanDistance(position, (w, h))

                if (not (walls[w][h])):
                    if (evidence >= 3):
                        # Absolute value because symetric distribution
                        noise = abs(evidence - distance)
                        # Probabilities come from the distribution Bin(4, 0.5)
                        if (noise == 0):
                            obser_matrix[w][h] = 6/16
                        elif (noise == 1):
                            obser_matrix[w][h] = 4/16
                        elif (noise == 2):
                            obser_matrix[w][h] = 1/16

                    elif (evidence == -1):
                        if (distance == 1):
                            obser_matrix[w][h] = 1
                    # All the probabilities have been normalize so that
                    # the sum equals 1.
                    elif (evidence == 0):
                        if (distance == 2):
                            obser_matrix[w][h] = 1/5
                        elif (distance == 1):
                            obser_matrix[w][h] = 4/5

                    elif (evidence == 1):
                        if (distance == 3):
                            obser_matrix[w][h] = 1/11
                        elif (distance == 2):
                            obser_matrix[w][h] = 4/11
                        elif (distance == 1):
                            obser_matrix[w][h] = 6/11

                    elif (evidence == 2):
                        if (distance == 4):
                            obser_matrix[w][h] = 1/15
                        elif (distance == 3):
                            obser_matrix[w][h] = 4/15
                        elif (distance == 2):
                            obser_matrix[w][h] = 6/15
                        elif (distance == 1):
                            obser_matrix[w][h] = 4/15

        return obser_matrix

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        T = self.transition_matrix(walls, position)
        OB = self.observation_matrix(walls, evidence, position)
        width = walls.width
        height = walls.height
        ghost_belief = np.zeros((width, height))

        # Quadruple loop because T is a four-dimensional matrix
        for i in range(width):
            for j in range(height):
                for k in range(width):
                    for m in range(height):
                        ghost_belief[i, j] += \
                            OB[i, j] * T[i, j, k, m] * belief[k, m]

        # Normalization of the updated ghost belief
        ghost_belief /= ghost_belief.sum()

        return ghost_belief

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(walls, beliefs[i],
                                             evidences[i], position)

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        pellets = list()

        for i in range(len(eaten)):
            if (not (eaten[i])):
                # List of pellets which have a "good" probability for a ghost
                good_probability = np.argwhere(beliefs[i] >= 0.99 *
                                               beliefs[i].max())

                for p in good_probability:
                    pellets.append((p[0], p[1]))

        direction = Directions.STOP
        min_heuristic = np.inf
        (w, h) = position
        neighbours = ((w - 1, h, Directions.WEST),
                      (w + 1, h, Directions.EAST),
                      (w, h - 1, Directions.SOUTH),
                      (w, h + 1, Directions.NORTH))

        # Research of the best move to do according to the
        # result of the heuristic function
        for (i, j, d) in neighbours:
            if (not (walls[i][j])):
                h = heuristic((i, j), pellets)

                if (h < min_heuristic):
                    min_heuristic = h
                    direction = d

        return direction

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
