class Env:
    def __init__(self, num_states, num_actions):
        """
        Base class for an environment.
        :param num_states: Number of states in the environment.
        :param num_actions: Number of actions in the environment.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.current_state = 0  # Default starting state

    def reset(self):
        """
        Reset the environment to the initial state.
        :return: Initial state.
        """
        self.current_state = 0
        return self.current_state

    def play(self, state, action):
        """
        Play a step in the environment.
        :param state: Current state.
        :param action: Action to take.
        :return: (next_state, reward)
        """
        raise NotImplementedError("This method should be implemented in subclasses.")


class Env1(Env):
    def __init__(self):
        super().__init__(num_states=2, num_actions=2)
        # Transition and reward dynamics for Env1
        self.transitions = {
            (0, 0): (0, 0),  # (state, action) -> (next_state, reward)
            (0, 1): (1, 0),
            (1, 0): (0, 0),
            (1, 1): (1, 1),
        }

    def play(self, state, action):
        return self.transitions.get((state, action), (state, 0))  # Default: no change


class Env2(Env):
    def __init__(self):
        super().__init__(num_states=3, num_actions=2)
        # Transition and reward dynamics for Env2
        self.transitions = {
            (0, 0): (0, 0),
            (0, 1): (1, 0),
            (1, 0): (2, 1),
            (1, 1): (0, 0),
            (2, 0): (1, 1),
            (2, 1): (2, 1),
        }

    def play(self, state, action):
        return self.transitions.get((state, action), (state, 0))  # Default: no change


class Env3(Env):
    def __init__(self):
        super().__init__(num_states=4, num_actions=2)
        # Transition and reward dynamics for Env3
        self.transitions = {
            (0, 0): (1, 0),
            (0, 1): (2, 0),
            (1, 0): (3, 0),
            (1, 1): (0, 0),
            (2, 0): (0, 0),
            (2, 1): (3, 1),
            (3, 0): (3, 0),
            (3, 1): (3, 0),
        }

    def play(self, state, action):
        return self.transitions.get((state, action), (state, 0))  # Default: no change


# Test the environments
if __name__ == "__main__":
    print("Testing Env1")
    env1 = Env1()
    state = env1.reset()
    print(f"Initial state: {state}")
    print(env1.play(0, 0))  # Expect (0, 0)
    print(env1.play(0, 1))  # Expect (1, 0)
    print(env1.play(1, 1))  # Expect (1, 1)

    print("\nTesting Env2")
    env2 = Env2()
    state = env2.reset()
    print(f"Initial state: {state}")
    print(env2.play(0, 1))  # Expect (1, 0)
    print(env2.play(1, 0))  # Expect (2, 1)

    print("\nTesting Env3")
    env3 = Env3()
    state = env3.reset()
    print(f"Initial state: {state}")
    print(env3.play(0, 0))  # Expect (1, 0)
    print(env3.play(1, 0))  # Expect (3, 0)
