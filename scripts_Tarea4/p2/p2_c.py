import numpy as np
import gymnasium as gym
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p2_functions import print_state_value_function

DESC = ["SFFFH",
        "FHFFF",
        "FFHFF",
        "HFFHF",
        "FFFFG"]

is_slippery = True
success_rate = 1/3
reward_schedule = (1, 0, 0)

env = gym.make(
    'FrozenLake-v1',
    desc=DESC,
    is_slippery=is_slippery,
    success_rate=success_rate,
    reward_schedule=reward_schedule
)

n_states = env.observation_space.n
n_actions = env.action_space.n
n_cols = 5

def greedy_action_with_random_tiebreak(Q_state):
    max_val = np.max(Q_state)
    candidates = np.flatnonzero(np.isclose(Q_state, max_val))
    return np.random.choice(candidates)

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        return greedy_action_with_random_tiebreak(Q[state])
def generate_episode(env, Q, epsilon):
    trajectory = []
    state, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = epsilon_greedy(Q, state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)

        trajectory.append((state, action, reward))
        total_reward += reward
        state = next_state 
        done = terminated or truncated

    return trajectory, total_reward, len(trajectory), done

def mc_first_visit(env, episodes, alpha=0.2, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000, gamma=0.99, debug=True):
    Q = np.zeros((n_states, n_actions))
    rewards_ep = []
    reached_goal_count = 0

    for ep in range(episodes):
        if epsilon_decay_steps > 0:
            eps = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_steps))
        else:
            eps = epsilon_start

        traj, tot_r, steps, done = generate_episode(env, Q, eps)
        rewards_ep.append(tot_r)
        if tot_r > 0:
            reached_goal_count += 1

        visited = set()
        G = 0.0
        for i in reversed(range(len(traj))):
            s, a, r = traj[i]
            G = r + gamma * G
            if (s, a) not in visited:
                visited.add((s, a))
                Q[s, a] += alpha * (G - Q[s, a])

        if debug and (ep < 100 or (ep+1) % (episodes//10) == 0):
            mean_recent = np.mean(rewards_ep[max(0, ep-100+1):ep+1])
            print(f"[FV] ep {ep+1}/{episodes}, eps={eps:.3f}, last_reward={tot_r:.2f}, mean100={mean_recent:.4f}, reached_goal_rate={(reached_goal_count/(ep+1))*100:.2f}%")

    return Q, rewards_ep

def mc_every_visit(env, episodes, alpha=0.2, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=10000, gamma=0.99, debug=True):
    Q = np.zeros((n_states, n_actions))
    rewards_ep = []
    reached_goal_count = 0

    for ep in range(episodes):
        if epsilon_decay_steps > 0:
            eps = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_steps))
        else:
            eps = epsilon_start

        traj, tot_r, steps, done = generate_episode(env, Q, eps)
        rewards_ep.append(tot_r)
        if tot_r > 0:
            reached_goal_count += 1

        G = 0.0
        for i in reversed(range(len(traj))):
            s, a, r = traj[i]
            G = r + gamma * G
            Q[s, a] += alpha * (G - Q[s, a])

        if debug and (ep < 100 or (ep+1) % (episodes//10) == 0):
            mean_recent = np.mean(rewards_ep[max(0, ep-100+1):ep+1])
            print(f"[EV] ep {ep+1}/{episodes}, eps={eps:.3f}, last_reward={tot_r:.2f}, mean100={mean_recent:.4f}, reached_goal_rate={(reached_goal_count/(ep+1))*100:.2f}%")

    return Q, rewards_ep

def plot_rewards(rewards, title, filename):
    plt.figure(figsize=(8,4))
    plt.plot(rewards, alpha=0.3, label='reward episodio')
    window = 100
    if len(rewards) >= window:
        mov = np.convolve(rewards, np.ones(window)/window, mode='valid')
        x = np.arange(window//2, window//2 + len(mov))
        plt.plot(x, mov, label=f"promedio movil ({window})")
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa total episodio')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

os.makedirs("results/exp_c", exist_ok=True)

EPISODES = 500000
ALPHA = 0.1
EPS_START = 0.3
EPS_END = 0.05
EPS_DECAY = EPISODES // 2

print("=== Ejecutando First-Visit MC ===")
Q_fv, rewards_fv = mc_first_visit(env, EPISODES, alpha=ALPHA, epsilon_start=EPS_START, epsilon_end=EPS_END, epsilon_decay_steps=EPS_DECAY, gamma=0.99, debug=True)

print("\n=== Ejecutando Every-Visit MC ===")
env2 = gym.make(
    'FrozenLake-v1',
    desc=DESC,
    is_slippery=is_slippery,
    success_rate=success_rate,
    reward_schedule=reward_schedule
)
Q_ev, rewards_ev = mc_every_visit(env2, EPISODES, alpha=ALPHA, epsilon_start=EPS_START, epsilon_end=EPS_END, epsilon_decay_steps=EPS_DECAY, gamma=0.99, debug=True)

V_fv = np.max(Q_fv, axis=1)
V_ev = np.max(Q_ev, axis=1)

print("\nMapa de V - First-Visit MC:")
print_state_value_function(V_fv, env.unwrapped.P, n_cols=n_cols, prec=4, title="FV-MC")

print("\nMapa de V - Every-Visit MC:")
print_state_value_function(V_ev, env.unwrapped.P, n_cols=n_cols, prec=4, title="EV-MC")

plot_rewards(rewards_fv, "First-Visit MC", "results/exp_c/rewards_fv.png")
plot_rewards(rewards_ev, "Every-Visit MC", "results/exp_c/rewards_ev.png")

np.save("results/exp_c/rewards_fv.npy", np.array(rewards_fv))
np.save("results/exp_c/rewards_ev.npy", np.array(rewards_ev))

print("\nResultados guardados en results/exp_c/")
