import gym
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

class OnePayoffMatrix(gym.Env):

    def __init__(self, payoff_matrix):

        self.payoff_matrix = payoff_matrix
        self.action_space = gym.spaces.Discrete(payoff_matrix.shape[0])
        self.observation_space = gym.spaces.Discrete(1)

        self.bounds = [(0, 1)] * self.payoff_matrix.shape[1]
        self.linear_constraint = LinearConstraint(
            np.ones(self.payoff_matrix.shape[1]), [1], [1]
        )

        self.reset()

    def reset(self):
        return self.get_state()
    
    def get_state(self):
        return [0]
        
    def _calculate_reward(self, agent_distribution, agent_actions):

        other_agent_action_space = self.payoff_matrix.shape[1]
        
        def expected_reward(action_dist, opponent_distribution, sign):
            exp = 0
            for i in range(self.action_space.n):
                exp += opponent_distribution[i] * np.sum(action_dist * sign * self.payoff_matrix[i])
            return -exp
 
        res = minimize(
            expected_reward, 
            np.ones(other_agent_action_space) / other_agent_action_space,
            args=(agent_distribution, -1),
            bounds=self.bounds,
            constraints=self.linear_constraint
        )

        opt_actions = np.random.choice(
            self.payoff_matrix.shape[1], 
            len(agent_actions), p=res.x
        )

        reward = 0
        for i in range(len(agent_actions)):
            reward += self.payoff_matrix[agent_actions[i]][opt_actions[i]]

        return reward

    
    def step(self, agent_distribution, agent_actions):
        rewards = self._calculate_reward(agent_distribution, agent_actions)
        return self.get_state(), rewards, True, [0]
