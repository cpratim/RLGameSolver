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

env = TwoPayOffMatrix(blotto_payoff_matrix)

agent1 = QLearningAgentLSTM(
    env.get_action_space(0), 
    env.observation_space.n, 
    LSTMBrain
)

agent2 = QLearningAgentLSTM(
    env.get_action_space(1),
    env.observation_space.n, 
    LinearBrain
)

# target_dist = np.array([4/9, 4/9, 0, 0, 1/9])
# target_dist = np.array([1/3, 1/3, 1/3])
# target_dist = np.array([0.16667,0.27778,0.55556])

def train(agent1, agent2, env, num_episodes=1000000, log_interval=1000, round_size=25):

    for episode in range(num_episodes):
        state = env.reset()
        done = False
    
        while not done:
            agent_1_distribution = agent1.get_agent_distribution(state)
            agent_2_distribution = agent2.get_agent_distribution(state)

            agent_1_actions = np.random.choice(
                env.get_action_space(0), 
                round_size, p=agent_1_distribution
            )
            agent_2_actions = np.random.choice(
                env.get_action_space(1),
                round_size, p=agent_2_distribution
            )
            
            _, reward, done, _ = env.step([agent_1_actions, agent_2_actions])

            states = [env.get_state()] * round_size
            agent1_loss = agent1.update_model(states, agent_1_actions, reward[0])
            agent2_loss = agent2.update_model(states, agent_2_actions, reward[1])

            # loss_items.append(loss)
            # average_reward.append(reward)
        
        if episode % log_interval == 0:
            print(f"Episode {episode}")
            print('Agent1 Dist:', agent_1_distribution, agent1_loss, reward[0])
            print('Agent2 Dist:', agent_2_distribution, agent2_loss, reward[1])


        # sleep(1)


if __name__ == '__main__':
    train(agent1, agent2, env)


    # dist = np.array([0.47740626, 0.33812416, 0.1844696 ]) 
    # reward = env._calculate_reward(target_dist)
    # random_dist = get_random_dist(5)
    # reward2 = env._calculate_reward(random_dist)
    
    # loss = -torch.clamp(torch.FloatTensor(target_dist), 1e-10, 1.0) * reward
    # loss = loss.sum()

    # loss2 = -torch.clamp(torch.FloatTensor(random_dist), 1e-10, 1.0) * reward2
    # print(random_dist, reward2, loss2.sum())
    # print(loss, reward)

    # reward = env._calculate_reward(target_dist)
    # agent_distribution = torch.FloatTensor(target_dist)
    # loss = torch.clamp(agent_distribution, 1e-10, 1.0) * reward 
    # loss = loss.sum() + 2
    
    # print(reward, loss)