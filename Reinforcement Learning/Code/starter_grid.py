import numpy as np
from Grid import Grid

# A function that samples a random action from the set of possible actions.
def random_action(grid):
    return np.random.choice(grid.get_possible_actions())
    
# Write a function random trajectory that takes a grid and a number of steps as input 
# and returns a list of states and a list of rewards.
def random_trajectory(grid, n_steps):
    states = []
    rewards = []
    for i in range(n_steps):
        action = random_action(grid)
        new_state, reward, game_over, _ = grid.move(action)
        states.append(new_state)
        rewards.append(reward)
        if game_over:
            break
    return states, rewards


# main program:
grid = Grid(5, 10)
print('___________________________')
grid.print()
print('___________________________')
id = grid._get_state()
print('state: ', id)
print('position: ', grid._id_to_position(id))
print('___________________________')
action = Grid.ACTION_UP
print('action: ', action)
new_state, reward, game_over, _ = grid.move(action)
print('new state: ', new_state)
print('new position: ', grid._id_to_position(new_state))
print('reward: ', reward)
print('___________________________')
grid.print()
print('___________________________')

states, rewards = random_trajectory(grid, 10)
print('states: ', states)
print('rewards: ', rewards)
print('___________________________')
