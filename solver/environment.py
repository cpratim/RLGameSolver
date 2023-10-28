import gym
from gym import spaces
import numpy as np


class MultiAgentEnv(gym.Env):

    def __init__(self, agents, game):
        self.agents = agents
        self.state = game.reset()
        self.action_space = spaces.Discrete(game.action_space_size)
        self.observation_space = spaces.Discrete(game.observation_space_size)
        self.n_agents = len(self.agents)
        self.agent_turn_idx = 0
        self.done = False

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.agent_turn_idx = 0
        self.done = False

    '''
    action:
        0: roll
        1: stop
        2: powerup_1
        3: powerup_2
    return:
        observation: np.array
        reward: float
        done: bool
        info: dict
    '''
    def step(self, action):
        agent = self.agents[self.agent_turn_idx]
        if action == 0:
            pass

    def _switch_agent_turn(self):
        self.agent_turn_idx = (self.agent_turn_idx + 1) % self.n_agents
