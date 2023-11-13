import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



class RPSGameEnv(gym.Env):
    def __init__(self, obs_size=0):
        super(RPSGameEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # 0: Rock, 1: Paper, 2: Scissors
        self.observation_space = gym.spaces.Discrete(1)  # No specific observation space for this example
        # self.agent1_last_actions = [0] * obs_size
        # self.agent2_last_actions = [0] * obs_size
        self.agent_wins = [0, 0, 0]
        self.ties = 0
        self.reset()

    def reset(self):
        # return self.agent1_last_actions + self.agent2_last_actions
        return [0]

    def step(self, actions):
     
        reward = self._calculate_reward(actions)
        # self.agent1_last_actions.pop(0)
        # self.agent1_last_actions.append(actions[0])
        # self.agent2_last_actions.pop(0)
        # self.agent2_last_actions.append(actions[1])

        # last_actions = self.agent1_last_actions + self.agent2_last_actions
        return 0, reward, True, [0]
    
    def _rps_logic(self, moves1, moves2):
        reward = 0

        for move1, move2 in zip(moves1, moves2):
            if move1 == move2:
                continue
            elif (move1 == 0 and move2 == 2) or (move1 == 1 and move2 == 0) or (move1 == 2 and move2 == 1):
                reward += 1
            else:
                reward -= 1
        return reward

    def _calculate_reward(self, actions, n_moves=1):

        # print(actions)
        agent_moves = [np.random.choice(3, n_moves, p=a) for a in actions]
        
        rewards = np.zeros(len(actions))
        for i in range(len(actions)):
            for j in range(len(actions)):
                if i == j:
                    continue
                agent1_reward = self._rps_logic(agent_moves[i], agent_moves[j])
                rewards[i] += agent1_reward
        return rewards


class LSTMAgent(nn.Module):
    def __init__(self, observation_space_size, action_space_size, lstm_units=10):
        super(LSTMAgent, self).__init__()
        self.lstm = nn.LSTM(observation_space_size, lstm_units)
        self.fc = nn.Linear(lstm_units, action_space_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)  # Take the last time step's output
        x = torch.abs(x) / torch.sum(torch.abs(x))  # Make sure the output sums to 1
        return x

# Q-learning agent with LSTM using PyTorch
class QLearningAgentLSTM:
    def __init__(self, action_space_size, observation_space_size, lstm_units=10):
        self.action_space_size = action_space_size
        self.model = LSTMAgent(observation_space_size, action_space_size, lstm_units)
        self.optimizer = optim.Adam(self.model.parameters(), lr=.001)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch and sequence dimensions
        action_probs = self.model(state)
        # return np.random.choice(self.action_space_size, p=action_probs.detach().numpy()[0])
        return action_probs.detach().numpy()[0]

    def update_model(self, state, reward):
        
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch and sequence dimensions

        action_probs = self.model(state)
        
        loss = -torch.log(torch.clamp(action_probs, 1e-10, 1.0)) * reward
        # loss = -torch.FloatTensor([reward,])
        # print(loss)

        loss = loss.max()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Training loop
env = RPSGameEnv()
lstm_units = 10

n_agents = 2

agents = [QLearningAgentLSTM(env.action_space.n, env.observation_space.n, lstm_units) for _ in range(n_agents)]

num_episodes = 100000
for episode in range(num_episodes):
    state = env.reset()
    done = False

    mean_loss = 0

    while not done:
        actions = [agent.choose_action(state) for agent in agents]
        _, rewards, done, _ = env.step(actions)
        loss = []
        for i in range(n_agents):
            l = agents[i].update_model(state, actions[i], rewards[i])
            loss.append(l)
        
        mean_loss = loss
        

    if episode % 10 == 0:
        print(f"Episode {episode}")
        dist = []
        for i in range(n_agents):
            dist.append(agents[i].choose_action(state))
        # dist = np.array(dist)
        print(np.mean(dist, axis=0), np.std(dist, axis=0))
        # print(mean_loss)

# Testing the trained agents
state = env.reset()
done = False

print("Testing the trained agents")

# while not done:cl
#     action1 = agent1.choose_action(state, log=True)
#     action2 = agent2.choose_action(state, log=True)
#     action3 = agent3.choose_action(state, log=True)

#     _, _, done, _ = env.step([action1, action2, action3])

print("Game Over")
