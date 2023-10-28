import numpy as np

class Entity():

    pass

class BaseGame():

    def __init__(self):
        self.agents = []
        self.state = None
        self.move = 0

    def step(self):
        pass


class World():

    def __init__(self, game):
        self.game = game 
        self.n_move = 0

    def step(self):

        pass
