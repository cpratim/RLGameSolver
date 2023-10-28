import numpy as np
from solver.core import BaseGame

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
    2. powerup_1: use powerup 1
    3. powerup_2: use powerup 2
'''


class DiceGamePlayer():

    def __init__(self, brain):
        self.brain = brain
        self.reward = 0
        self.reset()

    def get_signal(self, state):
        return self.brain.get_action(state)
    
    def get_state_vector(self):
        return np.array([
            self.turn_sum, int(self.used_powerup_1), int(self.used_powerup_2), int(self.rolled_6)
        ])
    
    def reset(self):
        self.used_powerup_1 = False
        self.used_powerup_2 = False
        self.rolled_6 = False
        self.turn_sum = 0


class DiceGame(BaseGame):

    def __init__(self, n_players):
        super().__init__()
        self.players = [DiceGamePlayer() for _ in range(n_players)]
        self.player_turn = 0
        self.signals = {
            0: 'roll',
            1: 'stop',
            2: 'powerup_1',
            3: 'powerup_2',
        }

    def _get_dice_roll(self):
        return np.random.randint(1, 7)
    
    def _translate(self, signal):
        return self.signals[signal]
    
    def reset(self):
        for player in self.players:
            player.reset()
        self.player_turn = 0

    def step(self):

        player = self.agents[self.player_turn]
        signal = player.get_signal(self.state.get_state_vector())
        signal = self.signals[signal]

        if player.roll_6:
            if signal == 'stop':
                player.turn_sum = 0
                player.roll_6 = False
            if signal == 'powerup_1':
                player.used_powerup_1 = True
                player.roll_6 = False
                player.score += player.turn_sum
                player.turn_sum = 0
                self.reward -= 3
            self.player_turn = (self.player_turn + 1) % len(self.agents)
        else:

            if signal == 'roll':
                roll = self._get_dice_roll()
                if roll == 6:
                    player.roll_6 = True
                player.turn_sum += roll
            if signal == 'stop':
                player.score += player.turn_sum
                player.turn_sum = 0
            if signal == 'powerup_2':
                for p in self.agents:
                    p.score += 10
                player.reward -= 2
            if player.score >= 30:
                player.reward += 10
                self.reset()
                
        