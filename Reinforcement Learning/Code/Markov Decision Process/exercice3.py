import numpy as np
from mdp import MDP
from agent import Agent

mdp = MDP(3, 2)

transitions = np.array([[[0., .8, .2], [.5, 0., .5]],
                        [[1., 0., 0.], [1., 0., 0.]],
                        [[0., 1., 0.], [0., 1., 0.]]])

rewards = np.array([[1.0, 0.], [2., 1.], [3., 2.]])
mdp.set_transitions(transitions)
mdp.set_rewards(rewards)

agent = Agent(3, 2)

policy = np.array(np.random.randint(2, size=3))
agent.set_policy(policy)

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

mdp.set_start_state(2)
n_steps = 10
total_reward, states, actions, rewards = mdp.simulate(agent, n_steps, trace=True)
print('Total reward:', total_reward)
for i in range(n_steps):
    print(f'{states[i]} -- {actions[i]} --> {states[i+1]}, {rewards[i]}')    