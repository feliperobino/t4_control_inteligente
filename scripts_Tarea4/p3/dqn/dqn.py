import gc
import logging
import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
import random
import numpy as np
import torch.nn as nn
from dqn.replay_memory import *

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



class Qnet(nn.Module):

    def __init__(self, hidden_size, n_observations, n_actions):
        super(Qnet, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class DQN(object):

    def __init__(self, gamma, tau, hidden_size, num_inputs, num_outputs, n_episodes=1000,
                 replay_memory_size=2000, batch_size=128):

        self.gamma = gamma
        self.tau = tau
        self.num_outputs = num_outputs
        self.n_episodes = n_episodes
        self.batch_size = batch_size

        self.Qnet = Qnet(hidden_size, num_inputs, self.num_outputs).to(device)
        self.Qnet_target = Qnet(hidden_size, num_inputs, self.num_outputs).to(device)

        # Define the optimizer
        self.qnet_optimizer = Adam(self.Qnet.parameters(), lr=3e-4)

        hard_update(self.Qnet_target, self.Qnet)
        #self.actions = np.linspace(-1, 1, num_outputs, dtype=np.float32)
        self.actions = np.linspace(0.0, self.I_max, self.num_outputs, dtype=np.float32)
        self.criterion = nn.SmoothL1Loss()
        self.replay_memory_size = replay_memory_size

        # Replay Memory
        self.memory = ReplayMemory(replay_memory_size)


    def calc_action(self, state, epsilon=0):
        # You must complete this method
        return 0, 0


    def collect_experience(self, state, action, done, next_state, reward):
        done = torch.Tensor([done])
        reward = torch.Tensor([reward])
        state = torch.Tensor([state]).reshape(1, -1)
        action = torch.Tensor([action]).reshape(1, -1)
        next_state = torch.Tensor([next_state]).reshape(1, -1)
        self.memory.push(state, action, done, next_state, reward)


    def learn(self, double_dqn=False, beta=0.4):
        self.qnet_optimizer.zero_grad()

        if len(self.memory) < self.batch_size:
            return 0

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.cat(batch.done).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)


        # State action values
        state_action_values = self.Qnet(state_batch).gather(1, action_batch.long()).squeeze()


        # Compute next states
        with torch.no_grad():
            next_state_action_values = self.Qnet_target(next_state_batch).max(1)[0]

        # Compute the target
        target = reward_batch + self.gamma * next_state_action_values*(1.0 - done_batch)


        # Calculate loss
        per_sample_loss = F.smooth_l1_loss(state_action_values, target, reduction='none')
        loss = per_sample_loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_value_(self.Qnet.parameters(), 100)
        self.qnet_optimizer.step()

        soft_update(self.Qnet_target, self.Qnet, self.tau)

        return loss.item()


    def set_eval(self):
        """
        Sets the model in evaluation mode

        """
        self.Qnet.eval()
        self.Qnet_target.eval()

    def set_train(self):
        """
        Sets the model in training mode

        """
        self.Qnet.train()
        self.Qnet_target.train()
