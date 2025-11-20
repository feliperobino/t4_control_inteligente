import gymnasium as gym
from p2_functions import  *
import matplotlib.pyplot as plt
import pandas as pd

env = gym.make(
    'FrozenLake-v1',
    desc=["SFFFH",
          "FHFFF",
          "FFHFF",
          "HFFHF",
          "FFFFG"],
    is_slippery=True,
    success_rate=1.0/3.0,
    reward_schedule=(1, 0, 0)
)

gamma = 0.99
P = env.unwrapped.P  # direct way to access the transition dynamics

n_cols = 5
n_episodes = 100000
optimal_Q, optimal_V, optimal_pi = value_iteration(P, gamma=gamma)
dicto_directions = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}
print('OPTIMAL PI')
for s, a in optimal_pi.items():
    print('s:{} | a {}'.format(s, dicto_directions[a]))

print_state_value_function(optimal_V, P, n_cols=n_cols, prec=4, title='Optimal state-value function:')
r_optimal = try_optimal_pi(optimal_pi, env, episodes=n_episodes)
