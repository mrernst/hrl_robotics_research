#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----

from util.replay_buffer import ReplayBuffer
import numpy as np
import torch
import gym
import argparse
import os
import sys

import imageio
import base64


# utilities
# -----
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


# custom classes
# -----

class Agent(object):
    def __init__(self, action_dim, policy, replay_buffer, burnin):
        self.action_dim = action_dim
        self._policy = policy
        self._replaybuffer = replay_buffer
        self.evaluations = []
        self._burnin = burnin
        # the agent should only get the policy model and then initialize it himself?

    def select_action(self, state, max_action, noise):
        """
        Select according to policy with random noise
        """
        action = (
            self._policy.select_action(np.array(state))
            + np.random.normal(0, max_action * noise, size=self.action_dim)
        ).clip(-max_action, max_action)

        return action

    def add_to_memory(self, state, action, next_state, reward, done):
        self._replaybuffer.add(state, action, next_state, reward, done)

    def learn(self, batch_size, tb_writer=None):
        self._policy.train(self._replaybuffer, batch_size, tb_writer)

        # write down metrics
        pass

    def reset(self):
        pass

    def eval_policy(self, env_name, seed, eval_episodes=10):
        """
        Runs policy for X episodes and returns average reward
        A fixed seed is used for the eval environment
        """
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            # state = np.concatenate([state[k] for k in state.keys()])
            while not done:
                action = self._policy.select_action(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                # state = np.concatenate([state[k] for k in state.keys()])
                avg_reward += reward

        avg_reward /= eval_episodes

        print(" " * 80 + "\r" + "---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        self.evaluations.append(avg_reward)
        pass

    def create_policy_eval_video(self, env_name, seed, filename, eval_episodes=5, fps=60):
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 100)

        filename = filename + ".mp4"
        with imageio.get_writer(filename, fps=fps) as video:
            for _ in range(eval_episodes):
                state, done = eval_env.reset(), False
                video.append_data(eval_env.render(mode='rgb_array'))
                while not done:
                    action = self._policy.select_action(np.array(state))
                    state, reward, done, _ = eval_env.step(action)
                    video.append_data(eval_env.render(mode='rgb_array'))
        pass


# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
