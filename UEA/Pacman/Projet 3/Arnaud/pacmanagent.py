# Complete this class for all parts of the project

from pacman_module.game import Agent
from pacman_module.pacman import Directions
import numpy as np
from pacman_module.util import manhattanDistance
from pacman_module.util import Queue
import math
from copy import deepcopy
import random


def compute_distance(state,pacman_position,w,h):
    """
    We create a real distance matrix from pacman_position 
    to all other cells to avoid loop due to walls
    """

    real_matrix = np.zeros((w,h))
    x = pacman_position[0]
    y = pacman_position[1]
    file = Queue()
    file.push([x,y])
    real_matrix[x,y] = 0
    while not file.isEmpty():
        
        case = file.pop()
        u = case[0]
        v = case [1]

        if not ((u+1)<0 or (u+1)>w-1 or v<0 or v>w-1):# a droite
            if real_matrix[u+1,v]==0:
                if not state.hasWall(u+1,v):
                    real_matrix[u+1,v] = real_matrix[u,v]+1
                    file.push([u+1,v])
                else:
                    real_matrix[u+1,v] = 10000
        
        if not ((u-1)<0 or (u-1)>w-1 or v<0 or v>w-1):#a gauche
            if real_matrix[u-1,v]==0:
                if not state.hasWall(u-1,v):
                    real_matrix[u-1,v] = real_matrix[u,v]+1
                    file.push([u-1,v])
                else:
                    real_matrix[u-1,v] = 10000
        
        if not (u<0 or u>w-1 or v+1<0 or v+1>w-1): # dessus
            if real_matrix[u,v+1]==0:
                if not state.hasWall(u,v+1):
                    real_matrix[u,v+1] = real_matrix[u,v]+1
                    file.push([u,v+1])
                else:
                    real_matrix[u,v+1] = 10000
        
        if not (u<0 or u>w-1 or v-1<0 or v-1>w-1):#dessous
            if real_matrix[u,v-1]==0:
                if not state.hasWall(u,v-1):
                    real_matrix[u,v-1] = real_matrix[u,v]+1
                    file.push([u,v-1])
                else:
                    real_matrix[u,v-1] = 10000
    return real_matrix



class PacmanAgent(Agent):
    def __init__(self, args):
        """
        Arguments:
        ----------
        - `args`: Namespace of arguments from command-line prompt.
        """
        self.args = args
        self.moves = []

    def get_action(self, state, belief_state):
        """
        Given a pacman game state and a belief state,
                returns a legal move.

        Arguments:
        ----------
        - `state`: the current game state. See FAQ and class
                   `pacman.GameState`.
        - `belief_state`: a list of probability matrices.

        Return:
        -------
        - A legal move as defined in `game.Directions`.
        """

        # XXX: Your code here to obtain bonus
        distance = []
        numberOfghost = len(belief_state)
        z = np.zeros(np.shape(belief_state[0]))
        mindis_ghost=1000
        actions = []
        number_of_possibleactions=0
        
        
        for ghost in range(numberOfghost):
            number_of_possibleactions=0
            w = np.shape(belief_state[ghost])[0]
            h = np.shape(belief_state[ghost])[1]
            if(np.sum(belief_state[ghost])!=0):
                coordx = 0
                coordy = 0
                vMax = 0
                for xx in range(w):
                    for yy in range(h):
                        if(belief_state[ghost][xx][yy]>vMax):
                            vMax = belief_state[ghost][xx][yy]
                            coordx = xx
                            coordy = yy
                coordVmax =  (coordx,coordy)

                for next_state,action in state.generatePacmanSuccessors():
                    actions.append(action)
                    number_of_possibleactions += 1
                    pacman_succ_position = next_state.getPacmanPosition()
                    matrice_real_distance = compute_distance(next_state,pacman_succ_position,w,h)
                    distance_real = matrice_real_distance[coordx,coordy]
                    if mindis_ghost > distance_real:
                        mindis_ghost = distance_real
                        action_doing = action
        

        random_number = random.randint(0, 10)
        if random_number == 0:
            if number_of_possibleactions >= 1:
                action_number = random.randint(0,number_of_possibleactions-1)
                action_doing = actions[action_number]

        return action_doing


        # XXX: End of your code here to obtain bonus




    
