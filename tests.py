import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Colonel Blotto payoff matrix
blotto_payoff_matrix = np.array([
    [4, 0, 2, 1],
    [0, 4, 1, 2],
    [1, -1, 3, 0],
    [-1, 1, 0, 3],
    [-2, -2, 2, 2]
])

for r in blotto_payoff_matrix:
    print(' '.join([str(x) for x in r]))

rock_paper_scissors_payoff_matrix = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0]
])

# Convert the NumPy matrix to a PyTorch tensor
payoff_matrix = torch.tensor(blotto_payoff_matrix, dtype=torch.float32)

# Neural Network architecture for the policy
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

# Training parameters
num_episodes = 25000
learning_rate = 0.01

# Initialize policies for both agents
input_size = len(payoff_matrix[0])
policy_agent1 = PolicyNetwork(input_size, payoff_matrix.shape[0])
policy_agent2 = PolicyNetwork(input_size, payoff_matrix.shape[1])

# Optimizers for both agents
optimizer_agent1 = optim.Adam(policy_agent1.parameters(), lr=learning_rate)
optimizer_agent2 = optim.Adam(policy_agent2.parameters(), lr=learning_rate)

# Training loop
for episode in range(num_episodes):
    # Sample actions from the policies
    actions_agent1 = torch.multinomial(policy_agent1(torch.ones(1, input_size)), 1)
    actions_agent2 = torch.multinomial(policy_agent2(torch.ones(1, input_size)), 1)

    # Get payoffs from the payoff matrix
    payoff_agent1 = payoff_matrix[actions_agent1.item(), actions_agent2.item()]
    payoff_agent2 = -payoff_matrix.T[actions_agent2.item(), actions_agent1.item()]

    # Calculate loss for both agents
    loss_agent1 = -torch.log(policy_agent1(torch.ones(1, input_size)))[0, actions_agent1.item()] * payoff_agent1
    loss_agent2 = -torch.log(policy_agent2(torch.ones(1, input_size)))[0, actions_agent2.item()] * payoff_agent2

    # Update policies
    optimizer_agent1.zero_grad()
    loss_agent1.backward()
    optimizer_agent1.step()

    optimizer_agent2.zero_grad()
    loss_agent2.backward()
    optimizer_agent2.step()

    # Print progress
    if (episode + 1) % 1000 == 0:
        print(payoff_agent1, payoff_agent2)
        print(f"Episode {episode + 1}/{num_episodes}, Loss Agent 1: {loss_agent1.item()}, Loss Agent 2: {loss_agent2.item()}")

# After training, you can use the learned policies for making decisions
# For example, you can sample actions from the policies:
sampled_actions_agent1 = torch.multinomial(policy_agent1(torch.ones(1, input_size)), 1).item()
sampled_actions_agent2 = torch.multinomial(policy_agent2(torch.ones(1, input_size)), 1).item()

# Output the final distribution for both agents
final_distribution_agent1 = torch.softmax(policy_agent1(torch.ones(1, input_size)), dim=-1).detach().numpy()
final_distribution_agent2 = torch.softmax(policy_agent2(torch.ones(1, input_size)), dim=-1).detach().numpy()

print("\nLearned Strategies:")
print("Agent 1 Strategy:", sampled_actions_agent1)
print("Agent 2 Strategy:", sampled_actions_agent2)

print("\nFinal Distribution for Agent 1:")
print(final_distribution_agent1)

print("\nFinal Distribution for Agent 2:")
print(final_distribution_agent2)
