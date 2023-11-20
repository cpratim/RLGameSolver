import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMBrain(nn.Module):

    '''
    A simple LSTM agent that takes in an observation and outputs an action distribution
    '''

    def __init__(self, observation_space_size, action_space_size, lstm_units=32):
        super(LSTMBrain, self).__init__()
        self.lstm = nn.LSTM(observation_space_size, lstm_units)
        self.fc = nn.Linear(lstm_units, action_space_size)

    '''
    Takes in an observation and outputs an action distribution
    '''
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x) 
        x = torch.abs(x) / torch.sum(torch.abs(x)) 
        return x
    

class LinearBrain(nn.Module):

    '''
    A simple linear agent that takes in an observation and outputs an action distribution
    '''

    def __init__(self, observation_space_size, action_space_size):
        super(LinearBrain, self).__init__()
        self.fc = nn.Linear(observation_space_size, action_space_size)

    '''
    Takes in an observation and outputs an action distribution
    '''
    def forward(self, x):
        x = self.fc(x) 
        x = torch.abs(x) / torch.sum(torch.abs(x)) 
        return x
    

class RNNBrain(nn.Module):

    '''
    A simple RNN agent that takes in an observation and outputs an action distribution
    '''
    def __init__(self, observation_space_size, action_space_size, rnn_units=8):
        super(RNNBrain, self).__init__()
        self.rnn = nn.RNN(observation_space_size, rnn_units)
        self.fc = nn.Linear(rnn_units, action_space_size)

    '''
    Takes in an observation and outputs an action distribution
    '''
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x) 
        x = torch.abs(x) / torch.sum(torch.abs(x)) 
        return x
    
    