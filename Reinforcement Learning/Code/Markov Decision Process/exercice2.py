import numpy as np
from agent import Agent
from mdp import MDP

mdp = MDP(2, 2)
agent = Agent(2, 2)
transitions = np.array([[[0.0, 1.0], [1.0, 0.0]], 
                        [[0.0, 1.0], [1.0, 0.0]]])
rawards = np.array([[1, 0], [2, 1]])
gamma = 0.9

mdp.set_transitions(transitions)
mdp.set_rewards(rawards)
mdp.set_gamma(gamma)
mdp.set_start_state(0)

agent.set_policy(np.array([0, 0, 0]))

mdp.set_start_state(0)
n_steps = 10
total_reward, states, actions, rewards = mdp.simulate(agent, n_steps, trace=True)
print('Total reward:', total_reward)
for i in range(n_steps):
    print(f'{states[i]} -- {actions[i]} --> {states[i+1]}, {rewards[i]}')

mdp.set_start_state(1)
n_steps = 10
total_reward, states, actions, rewards = mdp.simulate(agent, n_steps, trace=True)
print('Total reward:', total_reward)
for i in range(n_steps):
    print(f'{states[i]} -- {actions[i]} --> {states[i+1]}, {rewards[i]}')