#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class MetricLogger():
    def __init__(self, save_dir):
        self.writer = SummaryWriter(save_dir)

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Current episode metric
        self.init_episode()

        # episode counter
        self.episode_number = 1

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

        # write down reward at each step

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss /
                                   self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)
        self.episode_number += 1
        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
    
    def write_to_tensorboard(self, global_step):
        # extra function or even class to write stats down
        self.writer.add_scalar(
            'training/reward', self.ep_rewards[-1], global_step)
        self.writer.add_scalar(
            'training/episode_length', self.ep_lengths[-1], global_step)
        # self.writer.add_scalar('agent/avg_loss', ep_avg_loss, global_step)
        # self.writer.add_scalar('agent/avg_Q', ep_avg_q, global_step)
        