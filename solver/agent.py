import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QLearningAgentLSTM:

    def __init__(self, action_space_size, observation_space_size, solver_brain, learning_rate=.01):
        
        self.action_space_size = action_space_size
        learning_rate /= action_space_size
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

    def update_model(self, states, actions, reward):
        
        states = torch.FloatTensor(states)
        agent_distribution = self.model(states)

        # get total log likelihood of action sequence (i.e. product of likelihoods)
        log_likelihood = torch.log(torch.clamp(agent_distribution, 1e-10, 1.0))
        log_likelihood = log_likelihood[range(len(actions)), actions]
        log_likelihood = log_likelihood.sum()

        loss = -log_likelihood * reward
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
