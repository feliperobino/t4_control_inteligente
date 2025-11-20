import numpy as np
from tqdm import tqdm
import itertools
from tabulate import tabulate

def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi = {s:a for s, a in enumerate(np.argmax(Q, axis=1))}
    return Q, V, pi

def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")


def try_optimal_pi(pi, env, max_steps=200, episodes=2000):
    r_track = []

    for e in tqdm(range(episodes)):
        r_accum = 0
        state, info = env.reset()

        for t in range(max_steps):
            action = pi[state]
            next_state, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            state = next_state
            r_accum += reward
            if done:
                break
        r_track.append(r_accum)

    return np.array(r_track)




