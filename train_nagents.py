import numpy as np
from environment.multi_payoff_matrix import MultiPayoffMatrixEnv
from environment.one_payoff_matrix import TwoPayOffMatrix

from solver.agent import QLearningAgentLSTM
from solver.brain import LSTMBrain, LinearBrain, RNNBrain

# rock paper scissors matrices for 2 agents

blotto_payoff_matrix = np.array([
    [4, 0, 2, 1],
    [0, 4, 1, 2],
    [1, -1, 3, 0],
    [-1, 1, 0, 3],
    [-2, -2, 2, 2]
])

payoff_matrices = np.array([
    blotto_payoff_matrix, 
    -blotto_payoff_matrix
])

rock_paper_scissors_payoff_matrix = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0]
])

payoff_matrices = np.array([
    rock_paper_scissors_payoff_matrix, 
    -rock_paper_scissors_payoff_matrix
])

env = MultiPayoffMatrixEnv(payoff_matrices)
n_agents = payoff_matrices.shape[0]

agents = [
    QLearningAgentLSTM(
        env.get_action_space(i),
        env.observation_space.n,
        LSTMBrain,
    ) for i in range(n_agents)
]

def get_agent_distribution_static(agent_distributions):
    mean_dist = np.mean(agent_distributions, axis=0)
    var_dist = np.var(agent_distributions, axis=0)
    return mean_dist, var_dist


def train(agents, env, num_episodes=100000):
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            agent_distributions = [agent.get_agent_distribution(state) for agent in agents]
            _, rewards, done, _ = env.step(agent_distributions)
            for i in range(n_agents):
                agents[i].update_model(state, rewards[i])

        if episode % 1000 == 0:
            print(f"Episode {episode}")
            agent_distributions = [agent.get_agent_distribution(state) for agent in agents]
            # mean_dist, var_dist = get_agent_distribution_static(agent_distributions)
            # print('Mean distribution: ', mean_dist)
            # print('Variance distribution: ', var_dist)
            for agent in range(n_agents):
                print(f'Agent {agent} distribution: {list(agent_distributions[agent])}')


if __name__ == '__main__':

    train(agents, env)
