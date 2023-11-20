import gym
import numpy as np
import torch
# from scipy.optimize import differential_evolution
# from scipy.optimize import LinearConstraint

class MultiPayoffMatrixEnv(gym.Env):

    def __init__(self, payoff_matrices: np.ndarray, round_size=100):

        self.payoff_matrices = payoff_matrices
        self.action_space = None
        self.n_agents = payoff_matrices.shape[0]
        self.observation_space = gym.spaces.Discrete(1)
        self.round_size = round_size
        self.reset()

    def reset(self):
        return self.get_state()

    def get_action_space(self, agent_idx):
        return self.payoff_matrices[0].shape[agent_idx]
    
    def get_state(self):
        return torch.FloatTensor([0])
    
    # def add_random_noise()
    
    def _calculate_reward(self, agent_distributions):
        rewards_vec = np.zeros(self.n_agents)

        actions = np.array([
            np.random.choice(
                self.payoff_matrices[i].shape[i], 
                self.round_size, p=agent_distributions[i]
            ) for i in range(self.n_agents)
        ])

        for action_vec in actions.T:
            for agent in range(self.n_agents):
                rewards_vec[agent] += self.payoff_matrices[agent][tuple(action_vec)]

        return rewards_vec
    
    # def _calculate_reward_opt(self, agent_distributions):

    #     for agent in range(self.n_agents):
    #         action_dist = agent_distributions[agent]
    #         other_agent_action_space = self.get_action_space((agent + 1) % self.n_agents)
    #         bounds = [(0, 1)] * other_agent_action_space
    #         linear_constraint = LinearConstraint(np.ones(other_agent_action_space), [1], [1])

    #         def expected_reward(action_dist):
    #             exp = 0
    #             for i in range(other_agent_action_space):
    #                 exp += action_dist[i] * self.payoff_matrices[agent][i]


        
    def step(self, agent_distributions):

        rewards = self._calculate_reward(agent_distributions)
        return self.get_state(), rewards, True, [0]
