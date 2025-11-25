import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import sys

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

OUTDIR = "results/exp_d"
os.makedirs(OUTDIR, exist_ok=True)

env = gym.make(
    "FrozenLake-v1",
    desc=DESC,
    is_slippery=is_slippery,
    success_rate=success_rate,
    reward_schedule=reward_schedule
)

try:
    desc_arr = env.unwrapped.desc.astype(str)
    goal_index = None
    idx = 0
    for r in env.unwrapped.desc:
        for c in r:
            if c == ord('G'):
                goal_index = idx
            idx += 1
except Exception:
    goal_index = env.observation_space.n - 1

n_states = env.observation_space.n
n_actions = env.action_space.n
n_cols = int(np.sqrt(n_states))

print("Env states/actions:", n_states, n_actions, "goal_index:", goal_index)

def greedy_action_with_random_tiebreak(Q_state):
    max_val = np.max(Q_state)
    candidates = np.flatnonzero(np.isclose(Q_state, max_val))
    return np.random.choice(candidates)

def epsilon_greedy(Q, state, eps):
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    else:
        return greedy_action_with_random_tiebreak(Q[state])

def generate_episode_sarsa(env, Q, eps, max_steps=1000):
    traj = []
    obs, info = env.reset()
    s = obs
    done = False
    steps = 0
    total_reward = 0.0

    a = epsilon_greedy(Q, s, eps)
    while not done and steps < max_steps:
        s_next, r, terminated, truncated, info = env.step(a)
        if s_next == goal_index and r == 0:
            r = 1.0

        total_reward += r
        traj.append((s, a, r, s_next))
        done = terminated or truncated
        s = s_next
        if not done:
            a = epsilon_greedy(Q, s, eps)
        steps += 1

    return traj, total_reward, steps, done

def generate_episode_qlearn(env, Q, eps, max_steps=1000):
    traj = []
    obs, info = env.reset()
    s = obs
    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < max_steps:
        a = epsilon_greedy(Q, s, eps)
        s_next, r, terminated, truncated, info = env.step(a)
        if s_next == goal_index and r == 0:
            r = 1.0
        total_reward += r
        traj.append((s, a, r, s_next))
        done = terminated or truncated
        s = s_next
        steps += 1

    return traj, total_reward, steps, done

def run_sarsa(env, episodes=20000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=None, debug=True):
    Q = np.zeros((n_states, n_actions))
    rewards = []
    reach_count = 0
    epsilon = epsilon_start

    for ep in range(episodes):
        traj, tot_r, steps, done = generate_episode_sarsa(env, Q, epsilon)
        rewards.append(tot_r)
        if tot_r > 0:
            reach_count += 1

        for t in range(len(traj)):
            s, a, r, s_next = traj[t]
            if t < len(traj) - 1:
                a_next = traj[t+1][1]
            else:
                a_next = None

            target = r
            if a_next is not None:
                target += gamma * Q[s_next, a_next]
            Q[s, a] += alpha * (target - Q[s, a])

        if epsilon_decay is not None:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if debug and (ep < 50 or (ep+1) % (episodes//10) == 0):
            mean100 = np.mean(rewards[max(0, ep-100+1):ep+1])
            print(f"[SARSA] ep {ep+1}/{episodes}, eps={epsilon:.3f}, last_reward={tot_r:.2f}, mean100={mean100:.4f}, reach_rate={(reach_count/(ep+1))*100:.2f}%")

    return Q, rewards

def run_q_learning(env, episodes=20000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=None, debug=True):
    Q = np.zeros((n_states, n_actions))
    rewards = []
    reach_count = 0
    epsilon = epsilon_start

    for ep in range(episodes):
        traj, tot_r, steps, done = generate_episode_qlearn(env, Q, epsilon)
        rewards.append(tot_r)
        if tot_r > 0:
            reach_count += 1

        for (s, a, r, s_next) in traj:
            target = r + gamma * np.max(Q[s_next])
            Q[s, a] += alpha * (target - Q[s, a])

        if epsilon_decay is not None:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if debug and (ep < 50 or (ep+1) % (episodes//10) == 0):
            mean100 = np.mean(rewards[max(0, ep-100+1):ep+1])
            print(f"[Q-Learn] ep {ep+1}/{episodes}, eps={epsilon:.3f}, last_reward={tot_r:.2f}, mean100={mean100:.4f}, reach_rate={(reach_count/(ep+1))*100:.2f}%")

    return Q, rewards

def evaluate_greedy(Q, env, episodes=1000, render=False):
    total_returns = []
    for ep in range(episodes):
        obs, info = env.reset()
        s = obs
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < 1000:
            a = greedy_action_with_random_tiebreak(Q[s])
            s_next, r, terminated, truncated, info = env.step(a)
            if s_next == goal_index and r == 0:
                r = 1.0
            ep_ret += r
            s = s_next
            done = terminated or truncated
            steps += 1
        total_returns.append(ep_ret)
    return np.array(total_returns)

EPISODES = 20000
ALPHA = 0.1
GAMMA = 0.99

print("\n" + "="*60)
print("CONFIGURACIÓN DE EPSILON")
print("="*60)
print("Opciones:")
print("  [1] Epsilon fijo = 1.0 (exploración constante)")
print("  [2] Epsilon con decay (buenas prácticas)")
print("      - epsilon_start = 1.0")
print("      - epsilon_min = 0.05")
print("      - epsilon_decay = 0.995")
print("="*60)

use_decay = input("¿Usar epsilon con decay? [y/n]: ").strip().lower()

if use_decay == 'y':
    EPSILON_START = 1.0
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995
    print(f"\nUsando epsilon decay: {EPSILON_START} → {EPSILON_MIN} (decay={EPSILON_DECAY})\n")
else:
    EPSILON_START = 1.0
    EPSILON_MIN = 1.0
    EPSILON_DECAY = None
    print(f"\nUsando epsilon fijo = {EPSILON_START}\n")

print("CORRIENDO SARSA...")
Q_sarsa, rew_sarsa = run_sarsa(env, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, 
                                epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN, 
                                epsilon_decay=EPSILON_DECAY, debug=True)

# reiniciar env para Q-learning por limpieza
env_q = gym.make(
    "FrozenLake-v1",
    desc=DESC,
    is_slippery=is_slippery,
    success_rate=success_rate,
    reward_schedule=reward_schedule
)
print("\nCORRIENDO Q-LEARNING...")
Q_q, rew_q = run_q_learning(env_q, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, 
                              epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN, 
                              epsilon_decay=EPSILON_DECAY, debug=True)

V_sarsa = np.max(Q_sarsa, axis=1)
V_q = np.max(Q_q, axis=1)

print("\nMapa de V - SARSA:")
print_state_value_function(V_sarsa, env.unwrapped.P, n_cols=n_cols, prec=4, title="SARSA")

print("\nMapa de V - Q-learning:")
print_state_value_function(V_q, env.unwrapped.P, n_cols=n_cols, prec=4, title="Q-learning")

def plot_rewards(rewards, title, fname):
    plt.figure(figsize=(8,4))
    plt.plot(rewards, alpha=0.3, label="reward episodio")
    w = 100
    if len(rewards) >= w:
        mov = np.convolve(rewards, np.ones(w)/w, mode='valid')
        x = np.arange(w//2, w//2 + len(mov))
        plt.plot(x, mov, label=f"mov_avg ({w})")
    plt.title(title)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa total episodio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=150)
    plt.close()

plot_rewards(rew_sarsa, "SARSA", "rewards_sarsa.png")
plot_rewards(rew_q, "Q-learning", "rewards_q.png")

def save_V_grid(V, title, fname):
    grid = V.reshape((n_cols, n_cols))
    plt.figure(figsize=(5,5))
    plt.imshow(grid, interpolation='none')
    for (j,i), val in np.ndenumerate(grid):
        plt.text(i, j, f"{val:.3f}", ha='center', va='center')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=150)
    plt.close()

save_V_grid(V_sarsa, "V - SARSA", "V_sarsa.png")
save_V_grid(V_q, "V - Q-learning", "V_q.png")

EVAL_EP = 2000
print("\nEVALUANDO políticas aprendidas (greedy)...")
rets_sarsa = evaluate_greedy(Q_sarsa, env, episodes=EVAL_EP)
rets_q = evaluate_greedy(Q_q, env_q, episodes=EVAL_EP)

print(f"SARSA greedy eval: mean={rets_sarsa.mean():.4f}, std={rets_sarsa.std():.4f}")
print(f"Q-learning greedy eval: mean={rets_q.mean():.4f}, std={rets_q.std():.4f}")

np.save(os.path.join(OUTDIR, "rew_sarsa.npy"), np.array(rew_sarsa))
np.save(os.path.join(OUTDIR, "rew_q.npy"), np.array(rew_q))
np.save(os.path.join(OUTDIR, "eval_rets_sarsa.npy"), rets_sarsa)
np.save(os.path.join(OUTDIR, "eval_rets_q.npy"), rets_q)

print("\nTodos los resultados guardados en:", OUTDIR)
