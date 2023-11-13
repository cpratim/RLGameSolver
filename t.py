import numpy as np
from scipy.optimize import minimize, LinearConstraint, differential_evolution
from time import sleep

rock_paper_scissors_payoff_matrix = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0]
])

random_payoff_matrix = np.array([
    [10, 0, 0],
    [0, 6, 0],
    [0, 0, 3]
])

def get_random_dist(n):
    dist = np.random.rand(n)
    dist /= dist.sum()
    return dist

probability_constraint = LinearConstraint(
    np.ones(3), [1], [1]
)

def expected_reward(action_dist, agent_dist, payoff_matrix):

    exp = 0
    for i in range(payoff_matrix.shape[0]):
        exp += agent_dist[i] * np.sum(action_dist * -payoff_matrix[i])
    return -exp


def find_optimal_strageties(payoff_matrix):

    def other_agent_maximize(agent_dist):

        init_dist = get_random_dist(payoff_matrix.shape[1])
        res = minimize(
            expected_reward, 
            init_dist,
            args=(agent_dist, payoff_matrix),
            bounds=[(0, 1)] * payoff_matrix.shape[1], 
            constraints=probability_constraint
        )
        return -res.fun
    
    res = differential_evolution(
        other_agent_maximize,
        bounds=[(0, 1)] * payoff_matrix.shape[0],
        constraints=probability_constraint,
        maxiter=1000,
        disp=True,

    )
    return res.x, -res.fun    


res = find_optimal_strageties(random_payoff_matrix)
print(res)