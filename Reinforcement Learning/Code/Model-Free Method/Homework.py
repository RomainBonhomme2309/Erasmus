import random
import numpy as np
from collections import defaultdict


class MonteCarlo:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        """
        Initialize the Monte Carlo agent.
        :param env: The GridWorldMDP environment instance
        :param gamma: Discount factor
        :param epsilon: Exploration factor for epsilon-greedy exploration
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.V = {}
        self.N = {}
        self.policy = {} 

        for state in self.env.states:
            self.V[state] = 0
            self.N[state] = 0
            self.policy[state] = random.choice(self.env.actions)  # Random initial policy

    def epsilon_greedy_action(self, state):
        """
        Choose an action using epsilon-greedy exploration.
        :param state: Current state
        :return: Action to take
        """
        if random.random() < self.epsilon:  # Exploration
            return random.choice(self.env.actions)
        else:  # Exploitation
            return self.policy[state]

    def generate_episode(self):
        """
        Generate a single trajectory (episode) using the epsilon-greedy policy.
        :return: A list of (state, action, reward) tuples
        """
        trajectory = []
        state = self.env.initial_state
        done = False

        while not done:
            action = self.epsilon_greedy_action(state)
            next_state = self.get_next_state(state, action)
            reward = self.get_reward(state, action, next_state)
            trajectory.append((state, action, reward))
            state = next_state
            if state == self.env.terminal_state:
                done = True  # Terminate if we reach the terminal state

        return trajectory

    def get_next_state(self, state, action):
        """
        Returns the next state based on the current state and action.
        :param state: Current state
        :param action: Action taken
        :return: next_state
        """
        row, col = state
        action_row, action_col = action
        next_state = (row + action_row, col + action_col)

        # Ensure the next state is within bounds
        if 0 <= next_state[0] < self.env.height and 0 <= next_state[1] < self.env.width:
            return next_state
        else:
            return state

    def get_reward(self, state, action, next_state):
        """
        Get the reward for the transition (state, action, next_state).
        :param state: Current state
        :param action: Action taken
        :param next_state: Next state
        :return: Reward
        """
        return self.env.rewards.get((state, action, next_state), 0)

    def update_policy(self, num_episodes):
        """
        Perform Monte Carlo policy evaluation and update the policy.
        :param num_episodes: Number of episodes to run
        """
        for _ in range(num_episodes):
            trajectory = self.generate_episode()

            G = 0

            for state, _, reward in reversed(trajectory):
                G = reward + self.gamma * G
                self.N[state] += 1
                self.V[state] += (1 / self.N[state]) * (G - self.V[state])

            for state in self.policy:
                actions = self.env.actions
                best_action = max(actions, key=lambda a: self.evaluate_action_value(state, a))
                self.policy[state] = best_action

    def evaluate_action_value(self, state, action):
        """
        Helper to compute the action-value (Q-value) for a state-action pair.
        Approximate Q(s, a) using the current state-value function.
        """
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(state, action, next_state)
        return reward + self.gamma * self.V.get(next_state, 0)

    def display_value_function(self):
        """
        Print the state-value function in a grid format.
        """
        self.env.print_value_function(self.V)

    def display_policy(self):
        """
        Print the current policy in a grid format.
        """
        self.env.print_policy(self.policy)


class SARSAAgent:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        """
        Initialize the SARSA agent.
        :param env: The GridWorldMDP environment instance
        :param gamma: Discount factor
        :param alpha: Learning rate
        :param epsilon: Exploration factor for epsilon-greedy exploration
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.Q = {}
        for state in self.env.states:
            self.Q[state] = {}
            for action in self.env.actions:
                self.Q[state][action] = 0

        self.policy = {}
        for state in self.env.states:
            self.policy[state] = random.choice(self.env.actions)

    def select_action(self, state):
        """
        Choose an action using epsilon-greedy exploration.
        :param state: Current state
        :return: Action to take
        """
        if random.random() < self.epsilon:  # Exploration: choose a random action
            return random.choice(self.env.actions)
        else:  # Exploitation: choose the action with the highest Q-value
            return max(self.env.actions, key=lambda a: self.Q[state][a])

    def learn(self, num_episodes):
        """
        Perform SARSA learning over the specified number of episodes.
        :param num_episodes: Number of episodes to run
        """
        for _ in range(num_episodes):
            state = self.env.initial_state
            action = self.select_action(state)

            done = False
            while not done:
                next_state = self.get_next_state(state, action)
                reward = self.get_reward(state, action, next_state)

                next_action = self.select_action(next_state)

                # Update Q-value using the SARSA update rule
                self.Q[state][action] += self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])

                state = next_state
                action = next_action

                if state == self.env.terminal_state:
                    done = True

        self.update_policy()

    def get_next_state(self, state, action):
        """
        Returns the next state based on the current state and action.
        :param state: Current state
        :param action: Action taken
        :return: next_state
        """
        row, col = state
        action_row, action_col = action
        next_state = (row + action_row, col + action_col)

        # Ensure the next state is within bounds of the grid
        if 0 <= next_state[0] < self.env.height and 0 <= next_state[1] < self.env.width:
            return next_state
        else:
            return state  # Stay in place if the move goes out of bounds

    def get_reward(self, state, action, next_state):
        """
        Get the reward for the transition (state, action, next_state).
        :param state: Current state
        :param action: Action taken
        :param next_state: Next state
        :return: Reward
        """
        return self.env.rewards.get((state, action, next_state), 0)

    def update_policy(self):
        """
        Update the policy based on the learned Q-values.
        This method selects the best action (greedy policy) for each state.
        """
        for state in self.env.states:
            self.policy[state] = max(self.env.actions, key=lambda a: self.Q[state][a])

    def display_policy(self):
        """
        Print the learned policy in a grid format.
        """
        self.env.print_policy(self.policy)

    def display_value_function(self):
        """
        Print the learned state-action values in a grid format.
        """
        V = {}
        for state in self.env.states:
            V[state] = max(self.Q[state].values())  # Take the max Q-value for each state
        self.env.print_value_function(V)


class QLearningAgent:
    def __init__(self, env, gamma=0.9, alpha=0.1, epsilon=0.1):
        """
        Initialize the Q-learning agent.
        :param env: The GridWorldMDP environment instance
        :param gamma: Discount factor
        :param alpha: Learning rate
        :param epsilon: Exploration factor for epsilon-greedy exploration
        """
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.Q = {}
        for state in self.env.states:
            self.Q[state] = {}
            for action in self.env.actions:
                self.Q[state][action] = 0

        self.policy = {}
        for state in self.env.states:
            self.policy[state] = random.choice(self.env.actions)

    def select_action(self, state):
        """
        Choose an action using epsilon-greedy exploration.
        :param state: Current state
        :return: Action to take
        """
        if random.random() < self.epsilon:  # Exploration: choose a random action
            return random.choice(self.env.actions)
        else:  # Exploitation: choose the action with the highest Q-value
            return max(self.env.actions, key=lambda a: self.Q[state][a])

    def learn(self, num_episodes):
        """
        Perform Q-learning over the specified number of episodes.
        :param num_episodes: Number of episodes to run
        """
        for _ in range(num_episodes):
            state = self.env.initial_state
            done = False

            while not done:
                action = self.select_action(state)

                next_state = self.get_next_state(state, action)
                reward = self.get_reward(state, action, next_state)

                # Update Q-value using the Q-learning update rule
                max_next_q = max(self.Q[next_state].values())
                self.Q[state][action] += self.alpha * (reward + self.gamma * max_next_q - self.Q[state][action])


                state = next_state

                if state == self.env.terminal_state:
                    done = True

        self.update_policy()

    def get_next_state(self, state, action):
        """
        Returns the next state based on the current state and action.
        :param state: Current state
        :param action: Action taken
        :return: next_state
        """
        row, col = state
        action_row, action_col = action
        next_state = (row + action_row, col + action_col)

        # Ensure the next state is within bounds of the grid
        if 0 <= next_state[0] < self.env.height and 0 <= next_state[1] < self.env.width:
            return next_state
        else:
            return state  # Stay in place if the move goes out of bounds

    def get_reward(self, state, action, next_state):
        """
        Get the reward for the transition (state, action, next_state).
        :param state: Current state
        :param action: Action taken
        :param next_state: Next state
        :return: Reward
        """
        return self.env.rewards.get((state, action, next_state), 0)

    def update_policy(self):
        """
        Update the policy based on the learned Q-values.
        This method selects the best action (greedy policy) for each state.
        """
        for state in self.env.states:
            self.policy[state] = max(self.env.actions, key=lambda a: self.Q[state][a])

    def display_policy(self):
        """
        Print the learned policy in a grid format.
        """
        self.env.print_policy(self.policy)

    def display_value_function(self):
        """
        Print the learned state-action values in a grid format.
        """
        V = {}
        for state in self.env.states:
            V[state] = max(self.Q[state].values())  # State-value is the max Q-value for each state
        self.env.print_value_function(V)


if __name__ == "__main__":
    from grid_world import GridWorldMDP

    grid = GridWorldMDP(4, 4, 2)
    grid.print_board()

    print("Monte Carlo:")
    mc_agent = MonteCarlo(grid, gamma=0.9, epsilon=0.1)
    mc_agent.update_policy(num_episodes=10000)
    print("State-Value Function:")
    mc_agent.display_value_function()
    print("\nOptimal Policy:")
    mc_agent.display_policy()

    print("SARSA:")
    sarsa_agent = SARSAAgent(grid, gamma=0.95, alpha=0.1, epsilon=0.1)
    sarsa_agent.learn(num_episodes=10000)
    print("Learned Policy:")
    sarsa_agent.display_policy()
    print("\nState-Value Function:")
    sarsa_agent.display_value_function()

    print("\nQ-Learning:")
    qlearning_agent = QLearningAgent(grid, gamma=0.95, alpha=0.1, epsilon=0.1)
    qlearning_agent.learn(num_episodes=10000)
    print("Learned Policy:")
    qlearning_agent.display_policy()
    print("\nState-Value Function:")
    qlearning_agent.display_value_function()
