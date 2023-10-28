import gym
from gym import spaces
import numpy as np


class MultiAgentEnv(gym.Env):

    def __init__(self, game):
        self.state = game.reset()
        self.action_space = game.get_action_space()
        self.observation_space = game.get_observation_space()
        self.n_agents = game.get_n_agents()
        self.agent_turn_idx = 0
        self.move = 0

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
        agent_idx = self.agent_turn_idx
        updated_action_space, updated_observation_space = self.game.update(action, agent_idx)


    def _switch_agent_turn(self):
        self.agent_turn_idx = (self.agent_turn_idx + 1) % self.n_agents
