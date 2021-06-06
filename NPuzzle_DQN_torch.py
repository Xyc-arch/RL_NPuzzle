import math
import random
from os import stat
import random
import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.random import rand
np.random.seed(0)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from collections import namedtuple, deque
from itertools import count
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from tensorflow.python.util.nest import flatten
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from simulatorOrigin import NPUZZLE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Chart:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)
        # self.ax = self.ax.flatten()
        # self.ax1 = self.ax[0]
        # self.ax2 = self.ax[1]
        # self.ax2 = self.ax1.twinx()
        # plt.ion()
    
    def plot(self, episode_rewards, avg_loss):
        self.ax.clear()
        self.ax.plot(episode_rewards, color = 'r', label = '64 batch(64 sample/batch) reward')
        self.ax.set_xlabel('64 batch')
        self.ax.set_ylabel('64 batch(64 sample/batch) reward')
        plt.legend()
        self.fig.canvas.draw()

class Env:
    def __init__(self, simulator):
        self.simulator = simulator

    def reset(self, randomize=True, step_num = 10):
        simulator = self.simulator
        simulator.blank_loc = simulator.find_blank()
        if randomize: 
            for i in range(step_num):
                feasible_action = simulator.get_feasible_action(simulator.blank_loc)
                action = random.choice(feasible_action)
                simulator.step(action)
        simulator.org_mat = simulator.cur_mat.copy()
        return simulator.cur_mat

    def step(self, action):
        simulator = self.simulator
        simulator.step(action)
        next_state = simulator.cur_mat
        reward = - simulator.get_distance()
        done = simulator.is_done()
        return next_state, reward, done

    def feasible_action(self):
        simulator = self.simulator
        feasible_action = simulator.get_feasible_action(simulator.find_blank())
        return feasible_action


    def sample_action(self):
        feasible_action = self.feasible_action()
        feasible_action = torch.tensor(feasible_action)
        action = np.random.choice(feasible_action)
        return action
        
    def arg_max_q(self, torch_tensor):
        feasible_action = self.feasible_action()
        for i in range(4):
            if (i not in feasible_action):
                torch_tensor[i] = -1000
        values, indices = torch_tensor.topk(4)
        for i in range(4):
            if i != 0:
                if values[i] != values[i-1]:
                    break
            else:
                continue
        candidate = indices[0:i]
        action = random.choice(candidate)
        return action





class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                columns=['observation', 'action', 'reward',
                'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity
    
    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)
        
    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size**2, hidden_sizes[0]),
            nn.ReLU(),
            # nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            # nn.ReLU(),
            nn.Linear(hidden_sizes[0], output_size),
            nn.ReLU()
        )

    def forward(self, x):
        # x = x.reshape(x.shape[0], 9)
        if len(x.shape) <= 2:
            x = torch.unsqueeze(x, 0)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def weights_init(self, m): 
        if isinstance(m, nn.Conv2d): 
            nn.init.xavier_normal_(m.weight.data) 
            nn.init.xavier_normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias, 0)




class DQNAgent:
    def __init__(self, device, state_n, action_n, env, qnet_kwargs={}, gamma=0.99, epsilon=0.01,\
        replayer_capacity=100000, replayer_initial_transitions=5000, batch_size=64, lr=1e-3):
        
        self.device = device
        self.state_n = state_n
        self.action_n = action_n
        self.gamma = gamma
        self.epsilon = epsilon

        self.env = env
        self.batch_size = batch_size
        self.lr = lr
        self.replayer = DQNReplayer(replayer_capacity)
        self.eval_net = NeuralNetwork(self.env.simulator.grid_x, action_n, hidden_sizes=[60, 60])
        self.eval_net.apply(self.eval_net.weights_init)
        self.eval_net.to(self.device)
        self.target_net = NeuralNetwork(self.env.simulator.grid_x, action_n, hidden_sizes=[60, 60])
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.iteration_count = 0        
        self.loss = 0
        self.optimizer_eval_net = optim.Adam(self.eval_net.parameters(), lr=self.lr)

    def reset_env(self):
        self.iteration_count = 0
        self.loss = 0
        self.env.reset()
        

    def learn(self, state, action, reward, next_state, done):
        self.replayer.store(state, action, reward, next_state, done)
        
        states, actions, rewards, next_states, dones = self.replayer.sample(self.batch_size)
        batch_states = torch.FloatTensor(states).to(self.device)
        batch_actions = torch.from_numpy(actions).to(self.device)
        batch_rewards = torch.FloatTensor(rewards).to(self.device)
        batch_next_states = torch.FloatTensor(next_states).to(self.device)
        batch_dones = torch.from_numpy(dones).to(self.device)

        next_qs = self.target_net(batch_next_states)
        max_next_qs, _ = torch.max(next_qs, 1)
        us = batch_rewards + self.gamma * torch.logical_not(batch_dones) * max_next_qs
        targets = self.eval_net(batch_states)
        qs_pred = targets.clone()
        targets[np.arange(us.shape[0]), batch_actions] = us

        self.optimizer_eval_net.zero_grad()
        eval_net_loss = F.mse_loss(qs_pred, targets)
        self.loss = self.iteration_count * self.loss
        self.iteration_count += 1
        self.loss = (self.loss + eval_net_loss)/self.iteration_count
        eval_net_loss.backward()
        self.optimizer_eval_net.step()

        if done:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        
    def decide(self, state): # epsilon-greedy strategy
        if np.random.rand() < self.epsilon:
            return self.env.sample_action()
        state_tensor = torch.FloatTensor(state)
        state_tensor = state_tensor.to(device)
        with torch.no_grad():
            qs = self.eval_net(state_tensor)
        qs = torch.squeeze(qs, 0)
        action = self.env.arg_max_q(qs)
        return action.detach().cpu().numpy()       

    def save(self, save_dir, episode, num_64_batch):
        torch.save(self.eval_net, '{}/ql_policy_{}episode_{}64batch.pth'.format(save_dir, episode, num_64_batch))


def play_qlearning(agent, episode, train=False):
    env = agent.env
    all_iteration_reward = 0
    epoch_iteration_rewards = []
    chart = Chart()
    count = 0
    state = env.reset()
    while True:
        action = agent.decide(state)
        next_state, reward, done = env.step(action)
        count += 1
        all_iteration_reward += reward
        if train:
            agent.learn(state, action, reward, next_state,
                    done)
        if done:
            break
        avg_iteration_reward = all_iteration_reward/count
        if count % 4096 == 0:
            print('number_of_iteration = {}, average_loss = {}'.format(count, agent.loss))
            print(agent.env.simulator.cur_mat)
            if count % 8192 == 0:
                agent.save('./model', episode, count/4096)
            epoch_iteration_rewards.append(avg_iteration_reward)
            loss = agent.loss.detach().cpu().numpy()    
            chart.plot(epoch_iteration_rewards, loss)
            plt.savefig("/root/xuyichen/workspace/N_Puzzle/image/NPuzzle_rewards") 
        state = next_state
    # return episode_reward


def train_loop(agent, episodes):
    episode_num = 0
    for episode in range(episodes):
        if episode_num != 0:
            agent.reset_env()
        print('episode {} :'.format(episode + 1))
        play_qlearning(agent, episode, True)


simulator = NPUZZLE(3,3)
env = Env(simulator)
agent = DQNAgent(device, 3, 4, env)
train_loop(agent, 15)


        



    





