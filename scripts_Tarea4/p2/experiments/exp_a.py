import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from p2_functions import value_iteration, print_state_value_function, try_optimal_pi

DESC = ["SFFFH",
        "FHFFF",
        "FFHFF",
        "HFFHF",
        "FFFFG"]

MAP_NAME = None
is_slippery = True
reward_schedule = (1, 0, 0)
n_cols = 5
n_episodes = 5000

OUTDIR = os.path.join(os.getcwd(), "results", "exp_a")
os.makedirs(OUTDIR, exist_ok=True)

def make_env(success_rate=1.0/3.0, gamma=0.99):
    env = gym.make(
        'FrozenLake-v1',
        desc=DESC,
        is_slippery=is_slippery,
        success_rate=success_rate,
        reward_schedule=reward_schedule
    )
    return env

def run_vi_and_save(env, gamma, label):
    P = env.unwrapped.P
    Q, V, pi = value_iteration(P, gamma=gamma)

    # imprimir política y V
    print(f"\n--- Resultado: {label} ---")
    print("Política Óptima (por estado):")
    dicto_directions = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}
    for s, a in pi.items():
        print(f"s:{s:2d} -> {dicto_directions[a]}")
    print_state_value_function(V, P, n_cols=n_cols, prec=4, title=f'V (label={label}):')

    # guardamos V como imagen
    grid = V.reshape((n_cols, n_cols))
    plt.figure(figsize=(5,5))
    plt.imshow(grid, interpolation='none')
    for (j, i), val in np.ndenumerate(grid):
        plt.text(i, j, f"{val:.3f}", ha='center', va='center')
    plt.title(f'V grid - {label}')
    plt.colorbar()

    fname = os.path.join(OUTDIR, f'V_{label.replace(" ", "_")}.png')
    plt.savefig(fname, dpi=150)
    plt.close()

    # probar política por episodio y guardar estadisiticas
    r_track = try_optimal_pi(pi, env, episodes=n_episodes)
    np.save(os.path.join(OUTDIR, f'rewards_{label.replace(" ", "_")}.npy'), r_track)
    print(f"Recompensa media ({label}): {np.mean(r_track):.4f} +/- {np.std(r_track):.4f}")

    return Q, V, pi, r_track

##############
## Experimento A1: variar success_rate (gamma fijo = 0.99)
###############

gamma_fixed = 0.99
success_rates = [1.0/3.0, 1.0, 1.0/5.0]

for sr in success_rates:
    env = make_env(success_rate=sr, gamma=gamma_fixed)
    label = f"success_rate={sr:.2f}_gamma={gamma_fixed:.2f}"
    run_vi_and_save(env, gamma_fixed, label)

##############
## Experimento A2: variar gamma (success_rate fijo = 1/3)
###############

success_rate_fixed = 1.0/3.0
gammas = [0.99, 0.95]
for g in gammas:
    env = make_env(success_rate=success_rate_fixed, gamma=g)
    label = f"success_rate={success_rate_fixed:.2f}_gamma={g:.2f}"
    run_vi_and_save(env, g, label)

print(f'\n--- Experimento A finalizado === Salidas en {OUTDIR} ---\n')


def img_lake(desc):
    n = len(desc)
    grid = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if desc[i][j] == 'H':
                grid[i, j] = -1.0
            elif desc[i][j] == 'G':
                grid[i, j] = 1.0
            else:
                grid[i, j] = 0.0
    plt.figure(figsize=(5,5))
    plt.imshow(grid, cmap='coolwarm', interpolation='nearest')
    for (j, i), val in np.ndenumerate(grid):
        plt.text(i, j, f"{val:.1f}", ha='center', va='center')
    plt.title('Frozen Lake Layout')
    plt.colorbar()
    plt.show()

img_lake(DESC)