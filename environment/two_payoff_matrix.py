import gym
import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import LinearConstraint

class TwoPayOffMatrix(gym.Env):

    def __init__(self, payoff_matrix, round_size=10):

        self.payoff_matrix = payoff_matrix
        self.action_space = None
        self.observation_space = gym.spaces.Discrete(1)
        self.round_size = round_size
        self.reset()

    def get_action_space(self, agent_idx):
        return self.payoff_matrix.shape[agent_idx]

    def reset(self):
        return self.get_state()
 
    def get_state(self):
        return [0]
    

    def calculate_reward(self, agent_1_actions, agent_2_actions):

        rewards = np.zeros(2)

        for action_1, action_2 in zip(agent_1_actions, agent_2_actions):
            rewards[0] += self.payoff_matrix[action_1][action_2]
            rewards[1] += -self.payoff_matrix[action_1][action_2]

        rewards /= len(agent_1_actions)

        return rewards
        
    def step(self, agent_actions):
        rewards = self.calculate_reward(agent_actions[0], agent_actions[1])
        return self.get_state(), rewards, True, [0]


