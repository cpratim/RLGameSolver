import numpy as np
from environment.one_payoff_matrix import OnePayoffMatrix 
from solver.agent import QLearningAgentLSTM
from solver.brain import LinearBrain, RNNBrain
from warnings import filterwarnings
from matplotlib import pyplot as plt


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

target_dist_blotto = np.array([0.44444,0.44444,0,0,0.11111])

target_dist_rock_paper_scissors = np.array([0.33333,0.33333,0.33333])

target_dist_random = np.array([0.55,0.35,0.1])

def train(agent, env,target_dist, num_episodes=100000, log_interval=1000, tolerance=1e-3, round_size=5):

    episodes = []
    distances = []
    plt.figure()
    plt.title('Convergence of Agent Distribution to Target Distribution')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    line1, = plt.plot(episodes, distances, label='Agent')
    plt.legend()

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:

            agent_distribution = agent.get_agent_distribution(state)
            agent_actions = np.random.choice(
                env.payoff_matrix.shape[0],
                round_size, p=agent_distribution
            )

            states = [env.get_state()] * round_size
            _, reward, done, _ = env.step(agent_distribution, agent_actions)
            _ = agent.update_model(states, agent_actions, reward)

            
        distance = np.linalg.norm(target_dist- np.array(agent_distribution))
        if distance <= 0.001:
            break 
        episodes.append(episode)
        distances.append(distance)
        line1.set_data(episodes,distances)

        plt.xlim(0, episode)
        plt.ylim(0, max(distances))

        plt.draw()
        plt.pause(0.00000001)

        if episode % log_interval == 0:
            print(f"Episode {episode}")
            print(f'Agent distribution: {agent_distribution}')
            print("Distance: ", distance)
            
    plt.show()
      
if __name__ == '__main__':

    print("Choose a case (1(Rock Paper Scissors), 2(Colonel Blotto), or 3(Random Matrix) or enter anything else to input your own matrix and target):")
    case = int(input())

    if case == 1:
        payoff_matrix = rock_paper_scissors_payoff_matrix
        target_dist = target_dist_rock_paper_scissors

    elif case == 2:

        payoff_matrix = blotto_payoff_matrix
        target_dist = target_dist_blotto
    elif case == 3:

        payoff_matrix = random_payoff_matrix
        target_dist = target_dist_random
    else:
        rows = int(input("Enter the number of rows in the matrix: "))
        cols = int(input("Enter the number of columns in the matrix: "))
        
        print("Enter the elements of the 2D matrix (separate values with commas, press enter after each column):")
        payoff_matrix = np.array([list(map(float, input().split(','))) for _ in range(rows)])
        
        print("Enter the target distance vector (space-separated values):")
        target_dist = np.array(list(map(float, input().split())))
        
        if len(target_dist) != cols:
            print("Error: The size of the target distance vector should match the number of columns in the matrix.")
            exit()

    env = OnePayoffMatrix(payoff_matrix)

    agent = QLearningAgentLSTM(
        env.action_space.n, 
        env.observation_space.n, 
        LinearBrain
    )

    train(agent, env, target_dist)
    
    