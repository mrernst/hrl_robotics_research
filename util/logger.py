#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----

import numpy as np
import matplotlib.pyplot as plt
import torch
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
        self.episode_number = 0

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_steps += 1
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

        # write down reward at each step

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_steps)
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
        self.curr_ep_steps = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
    




def log_tensor_stats(tensor, name=None, writer=None, global_step=None, debug=False):
    """
    Takes a torch.tensor and returns name and shape for debugging purposes
    if writer is specified print_tensor_info writes basic statistics
    (min, max, std, mean) to tensorboard.
    """
    if debug:
        name = name if name else tensor.names
        text = "[DEBUG] name = {}, shape = {}, dtype = {}, device = {} \n" + \
            "\t min = {}, max = {}, std = {}, mean = {}"
        
        print(
            text.format(name, list(tensor.shape), tensor.dtype,
                        tensor.device.type, tensor.min(), tensor.max(),
                        tensor.type(torch.float).std(),
                        tensor.type(torch.float).mean()
                        )
                    )
    if writer:
        writer.add_scalar(
            f'{name}/min', tensor.min(),
            global_step=global_step)
        writer.add_scalar(
            f'{name}/max', tensor.max(),
            global_step=global_step)
        writer.add_scalar(
            f'{name}/std', tensor.type(torch.float).std(),
            global_step=global_step)
        writer.add_scalar(
            f'{name}/mean', tensor.type(torch.float).mean(),
            global_step=global_step)
    else:
        tensor_stat_dict = {
            f'{name}/min': tensor.min(),
            f'{name}/max': tensor.max(),
            f'{name}/std': tensor.type(torch.float).std(),
            f'{name}/mean': tensor.type(torch.float).mean(),
        }
        return tensor_stat_dict
        # return dict with values

    pass
