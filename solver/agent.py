import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QLearningAgentLSTM:

    def __init__(self, action_space_size, observation_space_size, solver_brain, learning_rate=.001):
        
        self.action_space_size = action_space_size
        self.model = solver_brain(observation_space_size, action_space_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def choose_action(self, state):

        state = torch.FloatTensor(state).unsqueeze(0)
        agent_distribution = self.model(state)
        return np.random.choice(
            self.action_space_size, 
            p=agent_distribution.detach().numpy()[0]
        )
    
    def get_agent_distribution(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        agent_distribution = self.model(state)
        return agent_distribution.detach().numpy()[0]

    def update_model(self, state, reward):
        
        state = torch.FloatTensor(state).unsqueeze(0)

        agent_distribution = self.model(state)
        loss = -torch.log(torch.clamp(agent_distribution, 1e-10, 1.0)) * reward

        loss = loss.max()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
