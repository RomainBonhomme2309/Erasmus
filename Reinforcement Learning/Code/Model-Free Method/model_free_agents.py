import random
import numpy as np
from collections import defaultdict

class MonteCarlo:
    def __init__(self, env, gamma=0.99):
        """
        Monte Carlo agent.
        :param env: The environment the agent interacts with.
        :param gamma: Discount factor.
        """
        self.env = env
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(env.num_actions))  # Action-value function
        self.returns = defaultdict(list)  # Store returns for state-action pairs
        self.policy = self.create_random_policy()

    def create_random_policy(self):
        """Creates a random policy."""
        return {state: random.choice(range(self.env.num_actions)) for state in range(self.env.num_states)}

    def generate_episode(self):
        """Generate an episode using the current policy."""
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            action = self.policy[state]
            next_state, reward = self.env.play(state, action)
            episode.append((state, action, reward))
            state = next_state
            if state == self.env.num_states - 1:  # Terminal state
                done = True
        return episode

    def update_policy(self, num_episodes=1000):
        """Update the policy using Monte Carlo."""
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            visited = set()
            for state, action, reward in reversed(episode):
                G = reward + self.gamma * G
                if (state, action) not in visited:
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])
                    visited.add((state, action))

            # Update policy to be greedy
            for state in self.policy.keys():
                self.policy[state] = np.argmax(self.Q[state])


class SARSAAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        SARSA agent.
        :param env: The environment the agent interacts with.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        :param epsilon: Exploration rate.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(env.num_actions))  # Action-value function

    def epsilon_greedy(self, state):
        """Epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(range(self.env.num_actions))
        return np.argmax(self.Q[state])

    def train(self, num_episodes=1000):
        """Train the agent using SARSA."""
        for _ in range(num_episodes):
            state = self.env.reset()
            action = self.epsilon_greedy(state)
            done = False
            while not done:
                next_state, reward = self.env.play(state, action)
                next_action = self.epsilon_greedy(next_state)
                # SARSA update
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action]
                )
                state, action = next_state, next_action
                if state == self.env.num_states - 1:  # Terminal state
                    done = True


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Q-learning agent.
        :param env: The environment the agent interacts with.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        :param epsilon: Exploration rate.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(env.num_actions))  # Action-value function

    def epsilon_greedy(self, state):
        """Epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.choice(range(self.env.num_actions))
        return np.argmax(self.Q[state])

    def train(self, num_episodes=1000):
        """Train the agent using Q-learning."""
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.epsilon_greedy(state)
                next_state, reward = self.env.play(state, action)
                # Q-learning update
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
                )
                state = next_state
                if state == self.env.num_states - 1:  # Terminal state
                    done = True
