import numpy as np
from solver.core import BaseGame
from gym import spaces

'''
Implement an environment for a dice game. The game is played by two players.
Each player has a score that starts at 0. The game is played in turns. In each
turn, a player rolls a dice. The number on the dice is added to the player's
score. The player can keep rolling the dice as long as they don't roll a 6.
If they roll a 6, they get 0 points for that turn and the turn ends. The first
player to reach 30 points or higher wins the game.

Each player has two powerups that they can use once in the game.
    1. the first powerup allows the player to keep their current sum if they roll a 6
    2. The second powerup allows the player to add 10 points to both their score and the opponent's score

Each powerup has an associated cost:
    1. The first powerup costs 3 points
    2. The second powerup costs 2 points

The player that wins gets 10 points 

The state of the game is represented by a dictionary with the following keys:
    1. player_1_score: the score of player 1
    2. player_2_score: the score of player 2
    3. current_player: the index of the current player

The action space is a dictionary with the following keys:
    1. roll: roll the dice
    2. stop: stop rolling the dice
    3. powerup_1: use powerup 1
    4. powerup_2: use powerup 2
'''


class DiceGameAgent():

    def __init__(self):
        self.reward = 0
        self.score = 0
        self.used_powerup_1 = False
        self.used_powerup_2 = False
        self.rolled_6 = False
        self.turn_sum = 0
        self.reset()
    
    def reset(self):
        self.used_powerup_1 = False
        self.used_powerup_2 = False
        self.rolled_6 = False
        self.turn_sum = 0

    def stop_turn(self, add_to_score=True):
        if add_to_score:
            self.score += self.turn_sum
        self.turn_sum = 0
        self.rolled_6 = False

    def use_powerup_1(self):
        self.used_powerup_1 = True
        self.reward -= 3

    def use_powerup_2(self):
        self.used_powerup_2 = True
        self.reward -= 2

    def roll(self):
        roll = self._roll()
        self.turn_sum += roll
        if roll == 6:
            self.rolled_6 = True
        return roll
    
    def get_action_space(self):
        space = [0, 1]
        if not self.used_powerup_1:
            space.append(2)
        if not self.used_powerup_2:
            space.append(3)
        return spaces.Discrete(len(space))
    
    def get_observation_space(self, on_turn=False):
        if on_turn:
            space = [int(self.rolled_6), self.turn_sum, self.score, self.used_powerup_1, self.used_powerup_2]
        else:
            space = [self.score, self.used_powerup_1, self.used_powerup_2]
        return space


class DiceGame(BaseGame):

    def __init__(self, n_agents):
        super().__init__()
        self.agents = [DiceGameAgent() for _ in range(n_agents)]
        self.n_agents = n_agents
        self.agent_turn_idx = 0
        self.move = 0

    def update(self, action, agent_idx):

        agent = self.agents[agent_idx]
        change_turn = False

        if agent.rolled_6:
            if agent.used_powerup_1:
                agent.stop_turn(add_to_score=False)
            elif action == 2:
                agent.use_powerup_1()
                agent.stop_turn(add_to_score=True)
            change_turn = True

        else:
            if action == 0:
                agent.roll()
            elif action == 1:
                agent.stop_turn(add_to_score=True)
                change_turn = True
            elif action == 3:
                if self.max_agent_score() < 20:
                    agent.use_powerup_2()
                    for agent in self.agents:
                        agent.score += 10
        if change_turn:
            self._switch_agent_turn()

        return self.get_observation_space(), self.get_action_space(), 

    def _switch_agent_turn(self):
        self.agent_turn_idx = (self.agent_turn_idx + 1) % self.n_agents

    def get_action_space(self):
        return self.agents[self.agent_turn_idx].get_action_space()

    def get_observation_space(self):
        space = self.agents[self.agent_turn_idx].get_observation_space(on_turn=True)
        for i in range(1, self.get_n_agents):
            idx = (self.agent_turn_idx + i) % self.get_n_agents
            space += self.agents[idx].get_observation_space(on_turn=False)
        return spaces.MultiDiscrete(space)

    def max_agent_score(self):
        return max([agent.score for agent in self.agents])
    
    def reset(self):
        for agent in self.agents:
            agent.reset()
