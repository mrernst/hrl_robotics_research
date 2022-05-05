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
        max_action_sub,
        model_path,
        model_save_freq,
        start_timesteps,
        # meta agent arguments
        buffer_size_meta,
        batch_size_meta,
        actor_lr_meta,
        critic_lr_meta,
        actor_hidden_layers_meta,
        critic_hidden_layers_meta,
        expl_noise_meta,
        policy_noise_meta,
        noise_clip_meta,
        discount_meta,
        policy_freq_meta,
        tau_meta,
        buffer_freq,
        train_freq,
        reward_scaling,
        # sub agent arguments
        buffer_size_sub,
        batch_size_sub,
        actor_lr_sub,
        critic_lr_sub,
        actor_hidden_layers_sub,
        critic_hidden_layers_sub,
        expl_noise_sub,
        policy_noise_sub,
        noise_clip_sub,
        discount_sub,
        policy_freq_sub,
        tau_sub,
        ):
        
        # TODO: decouple buffer frequency from subgoal frequency at some point
        # I don't think these should be the same anymore
        
        self.subgoal = Subgoal(subgoal_dim)
        max_action_meta = torch.Tensor(self.subgoal.action_space.high * np.ones(subgoal_dim))
        # scale policy_noise and noise clip (should maybe be done outside of the class)
        policy_noise_meta *= float(self.subgoal.action_space.high[0])
        noise_clip_meta *= float(self.subgoal.action_space.high[0])
        
        self.model_save_freq = model_save_freq
    
        self.high_con = HighLevelController(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=subgoal_dim,
            max_action=max_action_meta,
            model_path=model_path,
            actor_lr=actor_lr_meta,
            critic_lr=critic_lr_meta,
            actor_hidden_layers=actor_hidden_layers_meta,
            critic_hidden_layers=critic_hidden_layers_meta,
            expl_noise=expl_noise_meta,
            policy_noise=policy_noise_meta,
            noise_clip=noise_clip_meta,
            discount=discount_meta,
            policy_freq=policy_freq_meta,
            tau=tau_meta,
            )
    
        self.low_con = LowLevelController(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            max_action=max_action_sub,
            model_path=model_path,
            actor_lr=actor_lr_sub,
            critic_lr=critic_lr_sub,
            actor_hidden_layers=actor_hidden_layers_sub,
            critic_hidden_layers=critic_hidden_layers_sub,
            expl_noise=expl_noise_sub,
            policy_noise=policy_noise_sub,
            noise_clip=noise_clip_sub,
            discount=discount_sub,
            policy_freq=policy_freq_sub,
            tau=tau_sub,
            )
        
        self.controllers = [self.low_con, self.high_con]
    
        self.replay_buffer_sub = LowLevelReplayBuffer(
            state_dim=state_dim,
            goal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size_sub,
            batch_size=batch_size_sub
            )
    
        self.replay_buffer_meta = HighLevelReplayBuffer(
            state_dim=state_dim,
            goal_dim=goal_dim,
            subgoal_dim=subgoal_dim,
            action_dim=action_dim,
            buffer_size=buffer_size_meta,
            batch_size=batch_size_meta,
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
        with torch.no_grad():
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
                    #n_sg = self.subgoal.action_space.sample()
                    n_sg = self._sample_random_subgoal(step, s, self.sg, n_s)
                else:
                    n_sg = self._choose_subgoal_with_noise(step, s, self.sg, n_s)
            else:
                n_sg = self._choose_subgoal(step, s, self.sg, n_s)
            self.n_sg = n_sg
    
        return a, r, n_s, done
    
    def append(self, step, s, a, n_s, r, d):
        self.sr = self.low_reward(s, self.sg, n_s)
    
        # Low Replay Buffer
        self.replay_buffer_sub.append(
            s, self.sg, a, n_s, self.n_sg, self.sr, float(d))
    
        # High Replay Buffer
        if _is_update(step, self.buffer_freq, rem=1):
            if len(self.buf[6]) == self.buffer_freq:
                self.buf[4] = s
                self.buf[5] = float(d)
                self.replay_buffer_meta.append(
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
            loss, td_error = self.low_con.train(self.replay_buffer_sub)
            losses.update(loss)
            td_errors.update(td_error)
    
            if global_step % self.train_freq == 0:
                loss, td_error = self.high_con.train(self.replay_buffer_meta, self.low_con)
                losses.update(loss)
                td_errors.update(td_error)
    
        return losses, td_errors
    
    def _sample_random_subgoal(self, step, s, sg, n_s):
        if step % self.buffer_freq == 0: # Should be zero
            sg = self.subgoal.action_space.sample()
        else:
            sg = self.subgoal_transition(s, sg, n_s)
        
        return sg
    
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
    
    def _evaluate_low(self, state, last_state, sg, last_subgoal):
        """
        evaluate how the current low level agent
        is doing with following subgoals.
        """
        desired = np.array(last_state[:sg.shape[0]]) + np.array(last_subgoal[:sg.shape[0]])
        actual = np.array(state[:sg.shape[0]])
        # get difference between where we want to go and what was actually reached
        # this tests the effectiveness of the LL agent
        
        # difference in euclidean space
        self.low_con.curr_train_metrics['state_reached_diff'] = np.linalg.norm(actual - desired)
        
        # get directional diff
        followed_subgoal = np.array(state[:sg.shape[0]]) - np.array(last_state[:sg.shape[0]])
        
        reshaped_last_subgoal = np.array(last_subgoal[:sg.shape[0]]).reshape(1, -1)
        reshaped_followed_subgoal = followed_subgoal.reshape(1, -1)
        self.low_con.curr_train_metrics['state_reached_direction_diff'] = torch.nn.CosineSimilarity(reshaped_followed_subgoal,
            reshaped_last_subgoal)[0][0]
        
        # see difference in subgoals
        if self.subgoal_position is None:
            self.subgoal_position = np.array(sg[:sg.shape[0]])
        else:
            self.prev_subgoal_position = self.subgoal_position
            self.subgoal_position = np.array(sg[:sg.shape[0]])
            # from the difference, compute magnitude and direction
            self.low_con.curr_train_metrics['subgoals_mag_diff'] = np.linalg.norm(self.subgoal_position - self.prev_subgoal_position)
        
            reshaped_prev_subgoal_position = self.prev_subgoal_position.reshape(1, -1)
            reshaped_subgoal_position = self.subgoal_position.reshape(1, -1)
            self.low_con.curr_train_metrics['subgoals_direction_diff'] = torch.nn.CosineSimilarity(reshaped_subgoal_position,
                reshaped_prev_subgoal_position)[0][0]

    def subgoal_transition(self, s, sg, n_s):
        """
        subgoal transition function, provided as input to the low
        level controller.
        """
        return s[:sg.shape[0]] + sg - n_s[:sg.shape[0]]
    
    def low_reward(self, s, sg, n_s):
        """
        reward function for low level controller.
        rewards the low level controller for getting close to the
        subgoals assigned to it.
        """
        abs_s = s[:sg.shape[0]] + sg
        rew = -np.linalg.norm(abs_s - n_s[:sg.shape[0]], 2)
        return rew
        #return -np.sqrt(np.sum((abs_s - n_s[:sg.shape[0]])**2))
    
    def end_step(self):
        self.episode_subreward += self.sr
        self.sg = self.n_sg
    
    def end_episode(self, episode, logger=None):
        # if logger: 
        #     # log
        #     #logger.write('reward/Intrinsic Reward', self.episode_subreward, episode)
        #     logger.add_scalar(
        #     'training/intrinsic_reward', self.episode_subreward, episode)
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
