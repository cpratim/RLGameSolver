import numpy as np
import torch

class BaseBrain():

    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_action(self, state):
        pass


class LSTMBrain(BaseBrain):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
    def build_model(self):
        pass
