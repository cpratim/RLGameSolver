import numpy as np
from environment.two_payoff_matrix import TwoPayOffMatrix 
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

random_payoff_matrix = np.array([
    [10, 0, 0],
    [0, 6, 0],
    [0, 0, 3]
])

def get_random_dist(n):
    dist = np.random.rand(n)
    dist /= dist.sum()
    return dist

env = TwoPayOffMatrix(random_payoff_matrix)

agent = QLearningAgentLSTM(
    env.action_space.n, 
    env.observation_space.n, 
    LSTMBrain
)

# target_dist = np.array([4/9, 4/9, 0, 0, 1/9])
# target_dist = np.array([1/3, 1/3, 1/3])
target_dist = np.array([1/4, 1/2, 1/4])

def train(agent, env, num_episodes=100000, log_interval=1000, tolerance=1e-3):

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        loss_items = []
        average_reward = []

        while not done:
            agent_distribution = agent.get_agent_distribution(state)
            _, reward, done, _ = env.step(agent_distribution)
            loss = agent.update_model(state, reward)
            loss_items.append(loss)
            average_reward.append(reward)
        
        if episode % log_interval == 0:
            print(f"Episode {episode}")
            agent_distribution = agent.get_agent_distribution(state)
            diff = np.linalg.norm(np.abs(agent_distribution - target_dist))
            print('Agent distribution:', agent_distribution)
            print('Agent Loss', np.mean(loss_items))
            print('Average Reward:', np.mean(average_reward))
            print('Distance from target:', np.abs(agent_distribution - target_dist))
            print()

            if diff < tolerance:
                print('Agent has converged')
                break
            
            loss_items = []
            average_reward = []
        # sleep(1)


if __name__ == '__main__':
    train(agent, env)
    # dist = np.array([0.47740626, 0.33812416, 0.1844696 ]) 
    # reward = env._calculate_reward(target_dist)
    # random_dist = get_random_dist(5)
    # reward2 = env._calculate_reward(random_dist)
    
    # loss = -torch.clamp(torch.FloatTensor(target_dist), 1e-10, 1.0) * reward
    # loss = loss.sum()

    # loss2 = -torch.clamp(torch.FloatTensor(random_dist), 1e-10, 1.0) * reward2
    # print(random_dist, reward2, loss2.sum())
    # print(loss, reward)

    reward = env._calculate_reward(target_dist)
    agent_distribution = torch.FloatTensor(target_dist)
    loss = torch.clamp(agent_distribution, 1e-10, 1.0) * reward
    
    print(reward, loss.sum())