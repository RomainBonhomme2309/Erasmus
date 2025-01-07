# Complete this class for all parts of the project

from pacman_module.game import Agent
import numpy as np
from pacman_module import util
from scipy.stats import binom
from pacman_module.util import manhattanDistance
import scipy.special


class BeliefStateAgent(Agent):
   
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args

        """
            Variables to use in 'update_belief_state' method.
            Initialization occurs in 'get_action' method.

            XXX: DO NOT MODIFY THE DEFINITION OF THESE VARIABLES
            # Doing so will result in a 0 grade.
        """

        # Current list of belief states over ghost positions
        self.beliefGhostStates = None

        # Grid of walls (assigned with 'state.getWalls()' method)
        self.walls = None

        # Hyper-parameters
        self.ghost_type = self.args.ghostagent
        self.sensor_variance = self.args.sensorvariance

        self.p = 0.5
        self.n = int(self.sensor_variance/(self.p*(1-self.p)))

        # XXX: Your code here
        self.row =1
        # XXX: End of your code

    def _get_sensor_model(self, pacman_position, evidence):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:pacman_position[0] >  (width-distance)
        -------
        The sensor model represented as a 2D numpy array of
        size [width, height].
        The element at position (w, h) is the probability
        P(E_t=evidence | X_t=(w, h))
        """
        
        width = self.walls.width
        height = self.walls.height
        sensor = np.zeros((width,height))
        for w in range(width):
            for h in range(height-1,-1,-1):
                distance = manhattanDistance(pacman_position,(w,h))
                evidence = int(evidence)
                noise = evidence - distance
                noise = int(noise + self.n*self.p)
                if noise in range(0,self.n+1) and (self.walls[w][h]==False): 
                    coef_bino = scipy.special.comb(self.n,noise)
                    proba = coef_bino * (self.p**(noise)) * ((1-self.p)**(self.n-noise))
                    sensor[w,h] = proba

        return sensor
        pass
    
    def _get_transition_model(self, pacman_position):
        """
        Arguments:
        ----------
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step

        Return:
        -------
        The transition model represented as a 4D numpy array of
        size [width, height, width, height].
        The element at position (w1, h1, w2, h2) is the probability
        P(X_t+1=(w1, h1) | X_t=(w2, h2))
        """
        width = self.walls.width
        height = self.walls.height
        type_of_ghost = self.ghost_type
        array = np.zeros((width,height,width,height))
        counter = 0

        if type_of_ghost == "confused":
            proba_away = 1/4
            proba_closer = 1/4
        if type_of_ghost =="afraid":
            proba_away =2/3
            proba_closer=1/3
        if type_of_ghost == "scared" :
            proba_away = 8/9
            proba_closer= 1/9

        for w2 in range(width):
            for h2 in range(height-1,-1,-1):
                if (self.walls[w2][h2]==False):
                    w1_d = w2 + 1
                    w1_g = w2 - 1
                    h1_b = h2 - 1
                    h1_h = h2 + 1
                    distance_initial  = manhattanDistance(pacman_position,(w2,h2))
                
                    if  (self.walls[w1_d][h2]==False) and (w1_d < width):
                        counter += 1
                        distance_final  = manhattanDistance(pacman_position,(w1_d,h2))
                        if distance_final < distance_initial:
                            array[w1_d,h2,w2,h2] = proba_closer
                        else:
                            array[w1_d,h2,w2,h2] = proba_away
                
                    if (self.walls[w1_g][h2] == False) and (w1_g > 0):
                        counter += 1
                        distance_final  = manhattanDistance(pacman_position,(w1_g,h2))
                        if distance_final < distance_initial:
                            array[w1_g,h2,w2,h2] = proba_closer
                        else:
                            array[w1_g,h2,w2,h2] = proba_away
                
                    if (self.walls[w2][h1_h]==False) and (h1_h < height):
                        counter += 1
                        distance_final  = manhattanDistance(pacman_position,(w2,h1_h))
                        if distance_final < distance_initial:
                            array[w2,h1_h,w2,h2] = proba_closer
                        else:
                            array[w2,h1_h,w2,h2] = proba_away
                
                    if self.walls[w2][h1_b]== False and (h1_b > 0):
                        counter += 1
                        distance_final  = manhattanDistance(pacman_position,(w2,h1_b))
                        if distance_final < distance_initial:
                            array[w2,h1_b,w2,h2] = proba_closer
                        else:
                            array[w2,h1_b,w2,h2] = proba_away
                    if counter > 0:
                            array[:,:,w2,h2] = array[:,:,w2,h2] * (1/np.sum(array[:,:,w2,h2]))
        return array
        pass

    def _get_updated_belief(self, belief, evidences, pacman_position,
            ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        and the previous belief states before receiving the evidences,
        returns the updated list of belief states about ghosts positions

        Arguments:
        ----------
        - `belief`: A list of Z belief states at state x_{t-1}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze.
               Matrices filled with zeros must be returned for eaten ghosts.
        """

        # XXX: Your code here
        number_of_ghost = len(belief)
        width = self.walls.width
        height = self.walls.height
        transition = self._get_transition_model(pacman_position)
        
        for i in range(number_of_ghost):
            previous_belief = belief[i]
            aux = np.zeros((width, height))
            if ghosts_eaten[i] == True:
                belief[i] = aux
                continue
            else:
                sensor = self._get_sensor_model(pacman_position,evidences[i])
                for w in range(width):
                    for h in range(height-1,-1,-1):
                        x = previous_belief[w,h]
                        for k in range(width):
                            for l in range(height-1,-1,-1):
                                aux[k,l] = aux[k,l] + (transition[k,l,w,h] * x)     
                for w in range(width):
                    for h in range(height-1,-1,-1):
                        previous_belief[w,h] = sensor[w,h] * aux[w,h]
            
            previous_belief = previous_belief *  (1/np.sum(previous_belief))
            belief[i] = previous_belief
            
        # XXX: End of your code
        return belief

    def update_belief_state(self, evidences, pacman_position, ghosts_eaten):
        """
        Given a list of (noised) distances from pacman to ghosts,
        returns a list of belief states about ghosts positions

        Arguments:
        ----------
        - `evidences`: list of distances between
          pacman and ghosts at state x_{t}
          where 't' is the current time step
        - `pacman_position`: 2D coordinates position
          of pacman at state x_{t}
          where 't' is the current time step
        - `ghosts_eaten`: list of booleans indicating
          whether ghosts have been eaten or not

        Return:
        -------
        - A list of Z belief states at state x_{t}
          as N*M numpy mass probability matrices
          where N and M are respectively width and height
          of the maze layout and Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        belief = self._get_updated_belief(self.beliefGhostStates, evidences,
                                          pacman_position, ghosts_eaten)
        self.beliefGhostStates = belief
        return belief

    def _get_evidence(self, state):
        """
        Computes noisy distances between pacman and ghosts.

        Arguments:
        ----------
        - `state`: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class `pacman.GameState`.


        Return:
        -------
        - A list of Z noised distances in real numbers
          where Z is the number of ghosts.

        XXX: DO NOT MODIFY THIS FUNCTION !!!
        Doing so will result in a 0 grade.
        """
        positions = state.getGhostPositions()
        pacman_position = state.getPacmanPosition()
        noisy_distances = []

        for pos in positions:
            true_distance = util.manhattanDistance(pos, pacman_position)
            noise = binom.rvs(self.n, self.p) - self.n*self.p
            noisy_distances.append(true_distance + noise)

        return noisy_distances

    def _record_metrics(self, belief_states, state):
        """
        Use this function to record your metrics
        related to true and belief states.
        Won't be part of specification grading.

        Arguments:
        ----------
        - state: The current game state s_t
                   where 't' is the current time step.
                   See FAQ and class pacman.GameState.
        - belief_states: A list of Z
           NM numpy matrices of probabilities
           where N and M are respectively width and height
           of the maze layout and Z is the number of ghosts.

        N.B. : [0,0] is the bottom left corner of the maze

        """
   
        

    def get_action(self, state):
        """
        Given a pacman game state, returns a belief state.

        Arguments:
        ----------
        - `state`: the current game state.
                   See FAQ and class `pacman.GameState`.

        Return:
        -------
        - A belief state.
        """

        """
           XXX: DO NOT MODIFY THAT FUNCTION !!!
                Doing so will result in a 0 grade.
        """
        # Variables are specified in constructor.
        if self.beliefGhostStates is None:
            self.beliefGhostStates = state.getGhostBeliefStates()
        if self.walls is None:
            self.walls = state.getWalls()

        evidence = self._get_evidence(state)
        newBeliefStates = self.update_belief_state(evidence,
                                                   state.getPacmanPosition(),
                                                   state.data._eaten[1:])
        print(self._record_metrics(self.beliefGhostStates, state))


        return newBeliefStates, evidence








