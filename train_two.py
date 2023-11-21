import numpy as np
from environment.two_payoff_matrix import TwoPayOffMatrix 
from solver.agent import QLearningAgentLSTM
from solver.brain import LSTMBrain, LinearBrain, RNNBrain
from warnings import filterwarnings
from time import sleep
import torch
import matplotlib.pyplot as plt

filterwarnings('ignore')

rock_paper_scissors_payoff_matrix = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0]
])

blotto_payoff_matrix = np.array([
    [4, 0, 2, 1],
    [0, 4, 1, 2],
    [1, -1, 3, 0],
    [-1, 1, 0, 3],
    [-2, -2, 2, 2]
])

random_payoff_matrix = np.array([
    [1, 2, -1],
    [2, -1, 4],
    [-1, 4, 3]
])

target_dist_blotto1 = np.array([0.44444,0.44444,0,0,0.11111])
target_dist_blotto2 = np.array([0.03333,0.07778,0.53333,0.35556])

target_dist_rock_paper_scissors1 = np.array([0.33333,0.33333,0.33333])
target_dist_rock_paper_scissors2 = np.array([0.33333,0.33333,0.33333])

target_dist_random1 = np.array([0.55,0.35,0.1])
target_dist_random2 = np.array([0.55,0.35,0.1])

def train(agent1, agent2, env, target_dist1,target_dist2, num_episodes=1000000, log_interval=1000, round_size=25):

    episodes = []
    distances1 = []
    distances2 = []
    plt.figure()
    line1, = plt.plot(episodes, distances1, label='Agent 1')
    line2, = plt.plot(episodes, distances2, label='Agent 2')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.legend()
    plt.title('Convergence of two Agents playing against each other to Target Distribution')

    runagent1 = True
    runagent2 = True
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
    
        while not done:
            agent_1_distribution = agent1.get_agent_distribution(state)
            agent_2_distribution = agent2.get_agent_distribution(state)

            if runagent1:
                agent_1_actions = np.random.choice(
                    env.get_action_space(0), 
                    round_size, p=agent_1_distribution
                )
            if runagent2:
                agent_2_actions = np.random.choice(
                    env.get_action_space(1),
                    round_size, p=agent_2_distribution
                )
            
            _, reward, done, _ = env.step([agent_1_actions, agent_2_actions])

            states = [env.get_state()] * round_size
            if runagent1:
                agent1_loss = agent1.update_model(states, agent_1_actions, reward[0])
            if runagent2:
                agent2_loss = agent2.update_model(states, agent_2_actions, reward[1])
            distance1 = np.linalg.norm(target_dist1- np.array(agent_1_distribution))
            print(distance1)
            distance2 = np.linalg.norm(target_dist2- np.array(agent_2_distribution))
            print(distance2)
            if distance1 < 0.01:
                runagent1 = False
            if distance2 < 0.01:
                runagent2 = False
            

        if not runagent1 and not runagent2:
            break
        episodes.append(episode)
        distances1.append(distance1)
        distances2.append(distance2)
        
        line1.set_data(episodes, distances1)
        line2.set_data(episodes, distances2)

        # Adjust the plot limits
        plt.xlim(0, episode)
        plt.ylim(0, max(max(distances1), max(distances2)))

        plt.draw()
        plt.pause(0.00000001)
        
        
        
        if episode % log_interval == 0:
            print(f"Episode {episode}")
            print('Agent1 Dist:', agent_1_distribution, agent1_loss, reward[0])
            print('Agent2 Dist:', agent_2_distribution, agent2_loss, reward[1])


        # sleep(1)
    plt.show()


if __name__ == '__main__':
   
    print("Choose a case (1(Rock Paper Scissors), 2(Colonel Blotto), or 3(Random Matrix) or enter anything else to input your own matrix and target):")
    case = int(input())

    if case == 1:
        payoff_matrix = rock_paper_scissors_payoff_matrix
        target_dist1 = target_dist_rock_paper_scissors1
        target_dist2 = target_dist_rock_paper_scissors2

    elif case == 2:

        payoff_matrix = blotto_payoff_matrix
        target_dist1 = target_dist_blotto1
        target_dist2 = target_dist_blotto2
    elif case == 3:

        payoff_matrix = random_payoff_matrix
        target_dist1 = target_dist_random1
        target_dist2 = target_dist_random2
    else:
        rows = int(input("Enter the number of rows in the matrix: "))
        cols = int(input("Enter the number of columns in the matrix: "))
        
        print("Enter the elements of the 2D matrix (separate values with commas, press enter after each column):")
        payoff_matrix = np.array([list(map(float, input().split(','))) for _ in range(rows)])
        
        print("Enter the target distance vector for P1 (space-separated values):")
        target_dist1 = np.array(list(map(float, input().split())))

       
        
        if len(target_dist1) != cols:
            print("Error: The size of the target distance vector should match the number of columns in the matrix.")
            exit()
        
        print("Enter the target distance vector for P2 (space-separated values):")
        target_dist2 = np.array(list(map(float, input().split())))

        if len(target_dist2) != cols:
            print("Error: The size of the target distance vector should match the number of columns in the matrix.")
            exit()
        
    
    env = TwoPayOffMatrix(payoff_matrix)
    
    agent1 = QLearningAgentLSTM(
    env.get_action_space(0), 
    env.observation_space.n, 
    LSTMBrain
    )

    agent2 = QLearningAgentLSTM(
    env.get_action_space(1),
    env.observation_space.n, 
    LSTMBrain
    )

    train(agent1, agent2, env, target_dist1, target_dist2)