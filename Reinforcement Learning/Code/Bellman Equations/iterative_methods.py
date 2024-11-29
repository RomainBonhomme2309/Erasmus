import pprint as pp
import random
from grid_world import GridWorldMDP

GAMMA = 0.9


def policy_evaluation(gridworld: GridWorldMDP, policy, num_iterations=100):
    value_policy = {state: 0 for state in gridworld.states}

    for _ in range(num_iterations):
        new_value_policy = value_policy.copy()
        for state in gridworld.states:
            if state == gridworld.terminal_state or state in gridworld.bad_states:
                continue  # Skip terminal and bad states
            action = policy[state]
            value = 0
            for new_state in gridworld.states:
                prob = gridworld.transition_probabilities.get((state, action, new_state), 0)
                reward = gridworld.rewards.get((state, action, new_state), 0)
                value += prob * (reward + GAMMA * value_policy[new_state])
            new_value_policy[state] = value
        value_policy = new_value_policy
    return value_policy

def value_iteration(gridworld: GridWorldMDP, num_iterations=100) -> dict:
    optimal_values = {state: 0 for state in gridworld.states}

    for _ in range(num_iterations):
        new_optimal_values = optimal_values.copy()
        for state in gridworld.states:
            if state == gridworld.terminal_state or state in gridworld.bad_states:
                continue  # Skip terminal and bad states
            max_value = float('-inf')
            for action in gridworld.actions:
                value = 0
                for new_state in gridworld.states:
                    prob = gridworld.transition_probabilities.get((state, action, new_state), 0)
                    reward = gridworld.rewards.get((state, action, new_state), 0)
                    value += prob * (reward + GAMMA * optimal_values[new_state])
                max_value = max(max_value, value)
            new_optimal_values[state] = max_value
        optimal_values = new_optimal_values
    return optimal_values

def get_policy_from_value_function(gridworld: GridWorldMDP, value_function: dict) -> dict:
    policy = {}
    for state in gridworld.states:
        if state == gridworld.terminal_state or state in gridworld.bad_states:
            policy[state] = None  # No action for terminal or bad states
            continue
        best_action = None
        max_value = float('-inf')
        for action in gridworld.actions:
            value = 0
            for new_state in gridworld.states:
                prob = gridworld.transition_probabilities.get((state, action, new_state), 0)
                reward = gridworld.rewards.get((state, action, new_state), 0)
                value += prob * (reward + GAMMA * value_function[new_state])
            if value > max_value:
                max_value = value
                best_action = action
        policy[state] = best_action
    return policy


if __name__ == "__main__":
    grid = GridWorldMDP(10, 4, 4)
    grid.print_board()

    random_policy = {state: random.choice(grid.actions) for state in grid.states}
    value_policy = policy_evaluation(grid, random_policy, num_iterations=100)
    grid.print_value_function(value_policy)

    optimal_values = value_iteration(grid, num_iterations=100)
    grid.print_value_function(optimal_values)

    optimal_policy = get_policy_from_value_function(grid, optimal_values)
    grid.print_policy(optimal_policy)
