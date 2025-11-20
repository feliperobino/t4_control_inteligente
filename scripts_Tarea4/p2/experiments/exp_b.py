import os
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from p2_functions import value_iteration, print_state_value_function, try_optimal_pi

DESC = ["SFFFH",
        "FHFFF",
        "FFHFF",
        "HFFHF",
        "FFFFG"]

n_cols = 5
success_rate = 1/3
gamma = 0.99
is_slippery = False   # FIJAR EN FALSE (B)
n_episodes = 5000

OUTDIR = os.path.join(os.getcwd(), "results", "exp_b")
os.makedirs(OUTDIR, exist_ok=True)

def make_env(reward_schedule):
    env = gym.make(
        'FrozenLake-v1',
        desc=DESC,
        is_slippery=is_slippery,
        success_rate=success_rate,
        reward_schedule=reward_schedule
    )
    return env

def run_and_save(env, reward_schedule, label):
    P = env.unwrapped.P
    Q, V, pi = value_iteration(P, gamma=gamma)

    # imprimir política y V
    print(f"\n--- Resultado: {label} ---")
    print_state_value_function(V, P, n_cols=n_cols, prec=4, title=f'V (label={label}):')

    # guardamos V como imagen
    grid = V.reshape((n_cols, n_cols))
    plt.imshow(grid, cmap='coolwarm', interpolation='nearest')
    for (j, i), val in np.ndenumerate(grid):
        plt.text(i, j, f"{val:.3f}", ha='center', va='center')
    plt.colorbar()
    plt.title(f'V grid - {label}')
    plt.savefig(os.path.join(OUTDIR, f'V_{label}.png'), dpi=150)
    plt.close()

    r_track = try_optimal_pi(pi, env, episodes=n_episodes)
    np.save(os.path.join(OUTDIR, f'rewards_{label}.npy'), r_track)

    print(f"Mean reward: {np.mean(r_track):.4f}, std: {np.std(r_track):.4f}")

    return Q, V, pi

########3
# b1: reward_schedule = (1, 0, 0)

env_b1 = make_env(reward_schedule=(1, 0, 0))
Q_b1, V_b1, pi_b1 = run_and_save(env_b1, reward_schedule=(1, 0, 0), label="reward_100")


########
# b2: reward_schedule = (1, 0, -1)
env_b2 = make_env(reward_schedule=(1, 0, -1))
Q_b2, V_b2, pi_b2 = run_and_save(env_b2, reward_schedule=(1, 0, -1), label="reward_10m1")

print("\n--- Experimento B finalizado === Salidas en results/exp_b/ ---", OUTDIR)
