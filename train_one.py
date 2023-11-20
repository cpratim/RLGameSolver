import numpy as np
from environment.one_payoff_matrix import OnePayoffMatrix 
from solver.agent import QLearningAgentLSTM
from solver.brain import LSTMBrain, LinearBrain, RNNBrain
from warnings import filterwarnings
from time import sleep
import torch

filterwarnings('ignore')

rock_paper_scissors_payoff_matrix = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0]
])

blotto_payoff_matrix = np.array([
    [4, 0, 2, 1],
    [0, 4, 1, 2],
    [1, -1, 3, 0],
    [-1, 1, 0, 3],
    [-2, -2, 2, 2]
])

random_payoff_matrix = np.array([
    [1, 2, -1],
    [2, -1, 4],
    [-1, 4, 3]
])


def get_random_dist(n):
    dist = np.random.rand(n)
    dist /= dist.sum()
    return dist

target_dist = np.array([0.16667,0.27778,0.55556])

def train(agent, env, num_episodes=100000, log_interval=1000, tolerance=1e-3, round_size=5):

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        loss_items = []
        average_reward = []

        while not done:

            agent_distribution = agent.get_agent_distribution(state)

            agent_actions = np.random.choice(
                env.payoff_matrix.shape[0],
                round_size, p=agent_distribution
            )

            states = [env.get_state()] * round_size
            # print(states)
            _, reward, done, _ = env.step(agent_distribution, agent_actions)
            loss = agent.update_model(states, agent_actions, reward)
            # loss_items.append(loss)
            # average_reward.append(reward)
        
        if episode % log_interval == 0:
            print(f"Episode {episode}")
            print(f'Agent distribution: {agent_distribution}')
        # sleep(1)


if __name__ == '__main__':

    env = OnePayoffMatrix(blotto_payoff_matrix)

    agent = QLearningAgentLSTM(
        env.action_space.n, 
        env.observation_space.n, 
        LinearBrain
    )


    train(agent, env)
    
    