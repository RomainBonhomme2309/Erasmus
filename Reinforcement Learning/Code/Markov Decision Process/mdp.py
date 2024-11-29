# Description: This file contains the class MDP to define a Markov Decision Process (MDP) 
# with a finite number of states and actions.
import numpy as np
from agent import Agent
class MDP:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.transitions = np.zeros((n_states, n_actions, n_states))
        self.rewards = np.zeros((n_states, n_actions))
        self.gamma = 0.9
        self.start_state = 0
        self.current_state = 0

    def set_transitions(self, transitions):
        self.transitions = transitions

    def set_rewards(self, rewards):
        self.rewards = rewards

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_start_state(self, start_state):
        self.start_state = start_state
        self.current_state = start_state

    def reset(self):
        self.current_state = self.start_state
        return self.current_state

    def get_transitions(self):
        return self.transitions

    def get_rewards(self):
        return self.rewards

    def get_gamma(self):
        return self.gamma

    def get_possible_actions(self, state):
        return [action for action in range(self.n_actions) if np.sum(self.transitions[state, action, :]) > 0]

    def play(self, action):
        next_state = np.random.choice(range(self.n_states), p=self.transitions[self.current_state, action, :])
        reward = self.rewards[self.current_state, action]
        self.current_state = next_state
        return next_state, reward

    def simulate(self, agent, n_steps, trace=False):
        total_reward = 0
        self.reset()
        if trace:
            states = [self.current_state]
            actions = []
            rewards = []
        for i in range(n_steps):
            action = agent.choose_action(self.current_state)
            next_state, reward = self.play(action)
            if trace:
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)
            total_reward += self.gamma**i * reward
        if trace:
            return total_reward, states, actions, rewards
        return total_reward
