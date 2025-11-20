import random
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward')
                        )


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.5):

        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, *args):
        max_priority = max(self.priorities, default=1.0)

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = Transition(*args)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):

        if len(self.memory) == 0:
            raise ValueError("Cannot sample from an empty memory.")

        # use only the filled part if not yet at full capacity
        if len(self.memory) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:len(self.memory)])

        # compute sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # choose indices according to probs
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # importance sampling weights
        N = len(self.memory)
        # w_i = (1 / (N * P(i)))^beta
        weights = (N * probs[indices]) ** (-beta)
        # normalize so that max weight = 1 for stability
        weights /= weights.max()

        # convert to float32 for PyTorch use later
        weights = weights.astype(np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, td_errors, eps=1e-5):

        td_errors = np.abs(td_errors) + eps  # avoid zero priority
        for idx, prio in zip(indices, td_errors):
            self.priorities[idx] = float(prio)

    def __len__(self):
        return len(self.memory)
