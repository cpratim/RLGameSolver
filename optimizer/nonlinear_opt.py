import numpy as np
from scipy.optimize import minimize, LinearConstraint, differential_evolution
from time import sleep

blotto_payoff_matrix = np.array([
    [4, 0, 2, 1],
    [0, 4, 1, 2],
    [1, -1, 3, 0],
    [-1, 1, 0, 3],
    [-2, -2, 2, 2]
])

payoff_matrix = blotto_payoff_matrix

def get_random_dist(n):
    dist = np.random.rand(n)
    dist /= dist.sum()
    return dist

probability_constraint_row = LinearConstraint(
    np.ones(blotto_payoff_matrix.shape[0]), [1], [1]
)

probability_constraint_col = LinearConstraint(
    np.ones(blotto_payoff_matrix.shape[1]), [1], [1]
)

def expected_reward(action_dist, agent_dist, payoff_matrix):
    exp = 0
    for i in range(payoff_matrix.shape[0]):
        exp += agent_dist[i] * np.sum(action_dist * -payoff_matrix[i])
    return -exp

def other_agent_maximize(agent_dist, dist=False):

    init_dist = get_random_dist(payoff_matrix.shape[1])
    res = minimize(
        expected_reward, 
        init_dist,
        args=(agent_dist, payoff_matrix),
        bounds=[(0, 1)] * payoff_matrix.shape[1], 
        constraints=probability_constraint_col
    )
    if dist:
        return list(res.x)
    return -res.fun


def find_optimal_strageties(payoff_matrix):    
    res = differential_evolution(
        other_agent_maximize,
        bounds=[(0, 1)] * payoff_matrix.shape[0],
        constraints=probability_constraint_row,
        maxiter=1000,
        disp=True,

    )
    return list(res.x), -res.fun    


player_1_dist, val = find_optimal_strageties(blotto_payoff_matrix)
player_2_dist = other_agent_maximize(player_1_dist, dist=True)

print(player_1_dist)
print(player_2_dist)

print(val)