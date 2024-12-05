from Ex1 import Env1, Env2, Env3
from model_free_agents import MonteCarlo, SARSAAgent, QLearningAgent

if __name__ == "__main__":

    print("Testing Monte Carlo on Env1")
    env1 = Env1()
    mc_agent = MonteCarlo(env1)
    mc_agent.update_policy(1000)
    print(mc_agent.policy)

    print("\nTesting SARSA on Env2")
    env2 = Env2()
    sarsa_agent = SARSAAgent(env2)
    sarsa_agent.train(1000)
    print(sarsa_agent.Q)

    print("\nTesting Q-Learning on Env3")
    env3 = Env3()
    q_learning_agent = QLearningAgent(env3)
    q_learning_agent.train(1000)
    print(q_learning_agent.Q)
