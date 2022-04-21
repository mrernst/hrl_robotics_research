#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----

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

from agent.base import Agent
from algo.td3 import HighLevelController, LowLevelController, TD3Controller
from util.replay_buffer import LowLevelReplayBuffer, HighLevelReplayBuffer, ReplayBuffer
from util.utils import Subgoal, _is_update

# custom classes
# -----

class HiroAgent(Agent):
    def __init__(
        self,
        state_dim,
        action_dim,
        goal_dim,
        subgoal_dim,
        max_action_low,
        start_timesteps,
        model_save_freq,
        model_path,
        buffer_size_high,
        buffer_size_low,
        batch_size_high,
        batch_size_low,
        buffer_freq,
        train_freq,
        reward_scaling,
        policy_freq_high,
        policy_freq_low):
    
        self.subgoal = Subgoal(subgoal_dim)
        max_action_high = torch.Tensor(self.subgoal.action_space.high * np.ones(subgoal_dim))
        self.model_save_freq = model_save_freq
    
        self.high_con = HighLevelController(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            max_action=max_action_high,
            model_path=model_path,
            policy_freq=policy_freq_high
            )
    
        self.low_con = LowLevelController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            max_action=max_action_low,
            model_path=model_path,
            policy_freq=policy_freq_low
            )
    
        self.replay_buffer_low = LowLevelReplayBuffer(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size_low,
            batch_size=batch_size_low
            )
    
        self.replay_buffer_high = HighLevelReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size_high,
            batch_size=batch_size_high,
            freq=buffer_freq
            )
    
        self.buffer_freq = buffer_freq
        self.train_freq = train_freq
        self.reward_scaling = reward_scaling
        self.episode_subreward = 0
        self.sr = 0
    
        self.buf = [None, None, None, 0, None, None, [], []]
        self.fg = np.array([0,0])
        self.sg = self.subgoal.action_space.sample()
    
        self.start_timesteps = start_timesteps
    
    def step(self, s, env, step, global_step=0, explore=False):
        ## Lower Level Controller
        if explore:
            # Take random action for start_timesteps
            if global_step < self.start_timesteps:
                a = env.action_space.sample()
            else:
                a = self._choose_action_with_noise(s, self.sg)
        else:
            a = self._choose_action(s, self.sg)
    
        # Take action
        obs, r, done, _ = env.step(a)
        n_s = obs['observation']
    
        ## Higher Level Controller
        # Take random action for start_training steps
        if explore:
            if global_step < self.start_timesteps:
                n_sg = self.subgoal.action_space.sample()
            else:
                n_sg = self._choose_subgoal_with_noise(step, s, self.sg, n_s)
        else:
            n_sg = self._choose_subgoal(step, s, self.sg, n_s)
        
        self.n_sg = n_sg
    
        return a, r, n_s, done
    
    def append(self, step, s, a, n_s, r, d):
        self.sr = self.low_reward(s, self.sg, n_s)
    
        # Low Replay Buffer
        self.replay_buffer_low.append(
            s, self.sg, a, n_s, self.n_sg, self.sr, float(d))
    
        # High Replay Buffer
        if _is_update(step, self.buffer_freq, rem=1):
            if len(self.buf[6]) == self.buffer_freq:
                self.buf[4] = s
                self.buf[5] = float(d)
                self.replay_buffer_high.append(
                    state=self.buf[0],
                    goal=self.buf[1],
                    action=self.buf[2],
                    n_state=self.buf[4],
                    reward=self.buf[3],
                    done=self.buf[5],
                    state_arr=np.array(self.buf[6]),
                    action_arr=np.array(self.buf[7])
                )
            self.buf = [s, self.fg, self.sg, 0, None, None, [], []]
    
        self.buf[3] += self.reward_scaling * r
        self.buf[6].append(s)
        self.buf[7].append(a)
    
    def train(self, global_step):
        losses = {}
        td_errors = {}
    
        if global_step >= self.start_timesteps:
            loss, td_error = self.low_con.train(self.replay_buffer_low)
            losses.update(loss)
            td_errors.update(td_error)
    
            if global_step % self.train_freq == 0:
                loss, td_error = self.high_con.train(self.replay_buffer_high, self.low_con)
                losses.update(loss)
                td_errors.update(td_error)
    
        return losses, td_errors
    
    def _choose_action_with_noise(self, s, sg):
        return self.low_con.policy_with_noise(s, sg)
    
    def _choose_subgoal_with_noise(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0: # Should be zero
            sg = self.high_con.policy_with_noise(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)
    
        return sg
    
    def _choose_action(self, s, sg):
        return self.low_con.policy(s, sg)
    
    def _choose_subgoal(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0:
            sg = self.high_con.policy(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)
    
        return sg
    
    def subgoal_transition(self, s, sg, n_s):
        return s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]
    
    def low_reward(self, s, sg, n_s):
        abs_s = s[:sg.shape[0]] + sg
        return -np.sqrt(np.sum((abs_s - n_s[:sg.shape[0]])**2))
    
    def end_step(self):
        self.episode_subreward += self.sr
        self.sg = self.n_sg
    
    def end_episode(self, episode, logger=None):
        if logger: 
            # log
            logger.write('reward/Intrinsic Reward', self.episode_subreward, episode)
    
            # Save Model
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)
    
        self.episode_subreward = 0
        self.sr = 0
        self.buf = [None, None, None, 0, None, None, [], []]
    
    def save(self, episode):
        self.low_con.save(episode)
        self.high_con.save(episode)
    
    def load(self, episode):
        self.low_con.load(episode)
        self.high_con.load(episode)

# class HiroAgent(object):
#     """
#     Hierarchical Agent HIRO
#     """
# 
#     def __init__(self, action_dim, sub_algorithm, meta_algorithm, replay_buffer, c_timesteps):
#         self.action_dim = action_dim
#         self.sub_algorithm = sub_algorithm
#         self.meta_algorithm = meta_algorithm
#         self.replaybuffer = replay_buffer
#         self.evaluations = []
# 
#         # c is the internal timescale for sampling subgoals
#         self._counter = c_timesteps
# 
#     def step(self, state, max_action, noise):
#         """
#         Select according to policy with random noise
#         """
# 
#         action = (
#             self.algorithm.select_action(np.array(state))
#             + np.random.normal(0, max_action * noise, size=self.action_dim)
#         ).clip(-max_action, max_action)
# 
#         return action
# 
#     def add_to_memory(self, state, action, next_state, reward, done):
#         self.replaybuffer.add(None, None, state, action,
#                               next_state, reward, done)
# 
#     def learn(self, batch_size):
#         self.algorithm.train(self.replaybuffer, batch_size)
#         pass
# 
#     def reset(self):
#         pass
# 
#     def eval_policy(self, env_name, seed, eval_episodes=10):
#         """
#         Runs policy for X episodes and returns average reward
#         A fixed seed is used for the eval environment
#         """
#         eval_env = gym.make(env_name)
#         eval_env.seed(seed + 100)
# 
#         avg_reward = 0.
#         for _ in range(eval_episodes):
#             state, done = eval_env.reset(), False
#             state = get_obs_array(state)
#             while not done:
#                 action = self.algorithm.select_action(np.array(state))
#                 state, reward, done, _ = eval_env.step(action)
#                 state = get_obs_array(state)
# 
#                 avg_reward += reward
# 
#         avg_reward /= eval_episodes
#         self.evaluations.append(avg_reward)
#         return avg_reward
# 
#     def create_policy_eval_video(self, env_name, seed, filename, eval_episodes=15, fps=60):
#         eval_env = gym.make(env_name)
#         eval_env.seed(seed + 100)
# 
#         filename = filename + ".mp4"
#         with imageio.get_writer(filename, fps=fps) as video:
#             for _ in range(eval_episodes):
#                 state, done = eval_env.reset(), False
#                 state = get_obs_array(state)
#                 video.append_data(eval_env.unwrapped.render(
#                     mode='rgb_array').copy())
#                 while not done:
#                     action = self.algorithm.select_action(np.array(state))
#                     state, reward, done, _ = eval_env.step(action)
#                     state = get_obs_array(state)
#                     video.append_data(eval_env.unwrapped.render(
#                         mode='rgb_array').copy())
#         pass


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
