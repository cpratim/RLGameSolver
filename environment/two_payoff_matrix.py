import gym
import numpy as np
import torch
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import LinearConstraint

class TwoPayOffMatrix(gym.Env):

    def __init__(self, payoff_matrix):

        self.payoff_matrix = payoff_matrix
        self.action_space = gym.spaces.Discrete(payoff_matrix.shape[0])
        self.observation_space = gym.spaces.Discrete(1)
        self.reset()

    def reset(self):
        return self.get_state()
    
    def get_state(self):
        return torch.FloatTensor([0])
    
    # def add_random_noise()    
    
    def _calculate_reward(self, agent_distribution):

        other_agent_action_space = self.payoff_matrix.shape[1]
        bounds = [(0, 1)] * other_agent_action_space
        linear_constraint = LinearConstraint(
            np.ones(other_agent_action_space), [1], [1]
        )

        def expected_reward(action_dist):
            exp = 0
            for i in range(self.action_space.n):
                exp += agent_distribution[i] * np.sum(action_dist * -self.payoff_matrix[i])
            return -exp
        
        # res = differential_evolution(
        #     expected_reward, 
        #     bounds=bounds, 
        #     constraints=linear_constraint
        # )
        res = minimize(
            expected_reward, 
            np.ones(other_agent_action_space) / other_agent_action_space,
            bounds=bounds, 
            constraints=linear_constraint
        )
        return -res.fun
        

    def step(self, agent_distributions):
        rewards = self._calculate_reward(agent_distributions)
        return self.get_state(), rewards, True, [0]
