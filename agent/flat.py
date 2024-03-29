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

from agent.base import Agent
from algo.td3 import TD3Controller
from util.replay_buffer import ReplayBuffer, PERReplayBuffer
from util.utils import _is_update


class FlatAgent(Agent):
    """
    Default flag Agent based on TD3.
    """
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        max_action,
        model_path,
        model_save_freq,
        start_timesteps,
        prio_exp_replay,
        buffer_size,
        batch_size,
        actor_lr,
        critic_lr,
        actor_hidden_layers,
        critic_hidden_layers,
        expl_noise,
        policy_noise,
        noise_clip,
        discount,
        policy_freq,
        tau):
    
        self.con = TD3Controller(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            max_action=max_action,
            model_path=model_path,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_hidden_layers=actor_hidden_layers,
            critic_hidden_layers=critic_hidden_layers,
            expl_noise=expl_noise,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            discount=discount,
            policy_freq=policy_freq,
            tau=tau
            )
        self.controllers = [self.con]
        
        if prio_exp_replay:
            self.replay_buffer = PERReplayBuffer(
                state_dim=state_dim,
                goal_dim=goal_dim,
                action_dim=action_dim,
                buffer_size=buffer_size,
                batch_size=batch_size,
                )
            # define type of error by passing an integer
            self.replay_buffer.per_error_type = prio_exp_replay
        else:
            self.replay_buffer = ReplayBuffer(
                state_dim=state_dim,
                goal_dim=goal_dim,
                action_dim=action_dim,
                buffer_size=buffer_size,
                batch_size=batch_size
                )
        self.model_save_freq = model_save_freq
        self.start_timesteps = start_timesteps
    
    def step(self, s, env, step, global_step=0, explore=False):
        """
        Step the simulation.
        params:
            s: current state (torch.Tensor)
            env: gym environment (gym.env)
            step: current time step (int)
            global_step: current time step (int) #TODO why is that?
            explore: exploration boolean (bool)
        return:
            a: chosen action (torch.Tensor)
            r: received reward (float)
            n_s: next state (torch.Tensor)
            done: done boolean (bool)
        """
        if explore:
            if global_step < self.start_timesteps:
                a = env.action_space.sample()
            else:
                a = self._choose_action_with_noise(s)
        else:
            a = self._choose_action(s)
        
        obs, r, done, _ = env.step(a)
        n_s = obs['observation']
    
        return a, r, n_s, done
    
    def append(self, step, s, a, n_s, r, d):
        """
        Append variables to the buffer.
        params:
            step: current time step (int)
            s: current state (torch.Tensor)
            a: current action (torch.Tensor)
            n_s: next state (torch.Tensor)
            r: reward (float)
            d: done (bool)
        return:
            None
        """
        self.replay_buffer.append(s, self.fg, a, n_s, r, d)
    
    def train(self, global_step):
        """
        Train the agent's controllers.
        params:
            global_step: current time step (int)
        return:
            losses: Dictionary of current losses (dict)
            td_errors: Dictionary of current TD Errors (dict)
        """
        if global_step >= self.start_timesteps:
            return self.con.train(self.replay_buffer)
    
    def _choose_action(self, s):
        """
        Choose action with controller.
        params:
            s: current state (torch.Tensor)
        return:
            a: action (torch.Tensor)
        """
        return self.con.policy(s, self.fg)
    
    def _choose_action_with_noise(self, s):
        """
        Choose action with controller + noise.
        params:
            s: current state (torch.Tensor)
        return:
            a: action (torch.Tensor)
        """
        return self.con.policy_with_noise(s, self.fg)
    
    def end_step(self):
        """
        End the current step.
        params:
            None
        return:
            None
        """
        pass
    
    def end_episode(self, episode, logger=None):
        """
        End the current episode.
        params:
            episode: current episode (int)
            logger:
        return:
            None
        """
        if logger:
            if _is_update(episode, self.model_save_freq):
                self.save(episode=episode)
    
    def save(self, episode):
        """
        Save the agent's networks.
        params:
            episode: current episode (int)
        return:
            None
        """
        self.con.save(episode)
    
    def load(self, episode):
        """
        Load the agent's networks
        params:
            episode: episode number to load.
        return:
            None
        """
        self.con.load(episode)

# custom classes
# -----

# class Agent(object):
#     def __init__(self, action_dim, algorithm, replay_buffer):
#         self.action_dim = action_dim
#         self.algorithm = algorithm
#         self.replaybuffer = replay_buffer
#         self.evaluations = []
#         
#     def step(self, state, max_action, noise):
#         """
#         Select according to policy with random noise
#         """
#         action = (
#             self.algorithm.select_action(np.array(state))
#             + np.random.normal(0, max_action * noise, size=self.action_dim)
#         ).clip(-max_action, max_action)
# 
#         return action
# 
#     def add_to_memory(self, state, action, next_state, reward, done):
#         self.replaybuffer.add(None, None, state, action, next_state, reward, done)
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
#                 video.append_data(eval_env.unwrapped.render(mode='rgb_array').copy())
#                 while not done:
#                     action = self.algorithm.select_action(np.array(state))
#                     state, reward, done, _ = eval_env.step(action)
#                     state = get_obs_array(state)
#                     video.append_data(eval_env.unwrapped.render(mode='rgb_array').copy())
#         pass
# 
# 
# # _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
