#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----

import numpy as np
import torch
import torch.nn as nn
import gym
import argparse
import os
import sys

import imageio
import base64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# utilities
# -----
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from agent.base import Agent
from algo.td3 import HighLevelController, LowLevelController, TD3Controller
from util.replay_buffer import LowLevelReplayBuffer, HighLevelReplayBuffer, LowLevelPERReplayBuffer, HighLevelPERReplayBuffer, ReplayBuffer, PERReplayBuffer
from algo.state_compression import StateCompressor, SliceCompressor, EncoderCompressor, AutoEncoderCompressor, EncoderNetwork, AutoEncoderNetwork, SimCLR_TT_Loss, cosine_sim
from util.utils import Subgoal, AverageMeter, _is_update 


# custom classes
# -----

class HiroAgent(Agent):
    """
    Data-Efficient Hierarchical Reinforcement Learning Agent.
    """
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
        prio_exp_replay_meta,
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
        prio_exp_replay_sub,
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
        # I don't think these should be the same
        
        self.subgoal = Subgoal(subgoal_dim)
        max_action_meta = torch.Tensor(self.subgoal.action_space.high * np.ones(subgoal_dim))
        # scale policy_noise and noise clip (should maybe be done outside of the class)
        policy_noise_meta *= float(self.subgoal.action_space.high[0])
        noise_clip_meta *= float(self.subgoal.action_space.high[0])
        
        self.model_save_freq = model_save_freq
        

        self.state_compressor = SliceCompressor(subgoal_dim)
        
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
        
        if prio_exp_replay_sub:
            self.replay_buffer_sub = LowLevelPERReplayBuffer(
                state_dim=state_dim,
                goal_dim=subgoal_dim,
                action_dim=action_dim,
                buffer_size=buffer_size_sub,
                batch_size=batch_size_sub,
                )
            # define type of error by passing an integer
            self.replay_buffer_sub.per_error_type = prio_exp_replay_sub
        else:  
            self.replay_buffer_sub = LowLevelReplayBuffer(
                state_dim=state_dim,
                goal_dim=subgoal_dim,
                action_dim=action_dim,
                buffer_size=buffer_size_sub,
                batch_size=batch_size_sub,
                )
        
        if prio_exp_replay_meta:
            self.replay_buffer_meta = HighLevelPERReplayBuffer(
                state_dim=state_dim,
                goal_dim=goal_dim,
                subgoal_dim=subgoal_dim,
                action_dim=action_dim,
                buffer_size=buffer_size_meta,
                batch_size=batch_size_meta,
                freq=buffer_freq,
                )
            # define type of error by passing an integer
            self.replay_buffer_meta.per_error_type = prio_exp_replay_meta
        else:
            self.replay_buffer_meta = HighLevelReplayBuffer(
                state_dim=state_dim,
                goal_dim=goal_dim,
                subgoal_dim=subgoal_dim,
                action_dim=action_dim,
                buffer_size=buffer_size_meta,
                batch_size=batch_size_meta,
                freq=buffer_freq,
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
        """
        Train the agent's controllers.
        params:
            global_step: current time step (int)
        return:
            losses: Dictionary of current losses (dict)
            td_errors: Dictionary of current TD Errors (dict)
        """
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
        """
        Sample a random subgoal.
        params:
            step: current training step (int)
            s: current state (torch.Tensor)
            sg: current subgoal (torch.Tensor)
            n_s: next state (torch.Tensor)
        return:
            sg: random subgoal
        """
        if step % self.buffer_freq == 0: # Should be zero
            sg = self.subgoal.action_space.sample()
        else:
            sg = self.subgoal_transition(s, sg, n_s)
        
        return sg
    
    def _choose_action_with_noise(self, s, sg):
        """
        Choose action with low-level controller.
        params:
            s: current state (torch.Tensor)
            sg: current subgoal (torch.Tensor)
        return:
            a: action (torch.Tensor)
        """
        return self.low_con.policy_with_noise(s, sg)
    
    def _choose_subgoal_with_noise(self, step, s, sg, n_s):
        """
        Choose a subgoal via high-level controller + noise or transition existing subgoal.
        params:
            step: current training step (int)
            s: current state (torch.Tensor)
            sg: current subgoal (torch.Tensor)
            n_s: next state (torch.Tensor)
        returns:
            sg: next subgoal
        """
        if step % self.buffer_freq == 0: # Should be zero
            sg = self.high_con.policy_with_noise(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)
    
        return sg
    
    def _choose_action(self, s, sg):
        """
        Choose action with low-level controller.
        params:
            s: current state (torch.Tensor)
            sg: current subgoal (torch.Tensor)
        return:
            a: action (torch.Tensor)
        """
        return self.low_con.policy(s, sg)
    
    def _choose_subgoal(self, step, s, sg, n_s):
        """
        Choose a subgoal via high-level controller or transition existing subgoal.
        params:
            step: current training step (int)
            s: current state (torch.Tensor)
            sg: current subgoal (torch.Tensor)
            n_s: next state (torch.Tensor)
        returns:
            sg: next subgoal
        """
        if step % self.buffer_freq == 0:
            sg = self.high_con.policy(s, self.fg)
        else:
            sg = self.subgoal_transition(s, sg, n_s)
    
        return sg
    
    def _evaluate_low(self, state, last_state, sg, last_subgoal):
        """
        Evaluate how the current low level agent
        is doing with following subgoals.
        params:
            state: current state (torch.Tensor)
            last_state: the state some time steps before current state (torch.Tensor)
            sg: current subgoal freshly sampled (torch.Tensor)
            last_subgoal: the subgoal some time steps before sg (torch.Tensor)
        return:
            None
        """
        # state_reached_diff = AverageMeter()
        # state_reached_direction_diff = AverageMeter()
        # subgoals_mag_diff = AverageMeter()
        # subgoals_direction_diff = AverageMeter()
        
        
        # TODO: use an AverageMeter to keep track of the data in averages for logging
        # only write down statistics to the logs
        #desired = np.array(last_state[:sg.shape[0]]) + np.array(last_subgoal[:sg.shape[0]])
        #actual = np.array(state[:sg.shape[0]])
        desired = self.state_compressor.eval(last_state) + last_subgoal
        actual = self.state_compressor.eval(state)
        
        # get difference between where we want to go and what was actually reached
        # this tests the effectiveness of the LL agent
        
        # difference in euclidean space
        #self.low_con.curr_train_metrics['state_reached_diff'] = np.linalg.norm(actual - desired)
        state_reached_diff = torch.linalg.norm(actual - desired)
        
        # get directional diff
        #followed_subgoal = np.array(state[:sg.shape[0]]) - np.array(last_state[:sg.shape[0]])
        followed_subgoal = self.state_compressor.eval(state) - self.state_compressor.eval(last_state)
        
        #reshaped_last_subgoal = np.array(last_subgoal[:sg.shape[0]]).reshape(1, -1)
        #reshaped_followed_subgoal = followed_subgoal.reshape(1, -1)
        
        reshaped_last_subgoal = last_subgoal.reshape(1, -1)
        reshaped_followed_subgoal = followed_subgoal.reshape(1, -1)
        
        #self.low_con.curr_train_metrics['state_reached_direction_diff'] = torch.nn.CosineSimilarity(reshaped_followed_subgoal,
        #    reshaped_last_subgoal)[0][0]
        
        state_reached_direction_diff = torch.nn.CosineSimilarity(reshaped_followed_subgoal, reshaped_last_subgoal)[0][0]
        # see difference in subgoals
        
        if self.subgoal_position is None:
            self.subgoal_position = np.array(sg[:sg.shape[0]])
        else:
            self.prev_subgoal_position = self.subgoal_position
            self.subgoal_position = np.array(sg[:sg.shape[0]])
            # from the difference, compute magnitude and direction
            #self.low_con.curr_train_metrics['subgoals_mag_diff'] = np.linalg.norm(self.subgoal_position - self.prev_subgoal_position)
            subgoal_mag_diff = torch.linalg.norm(self.subgoal_position - self.prev_subgoal_position)
        
            reshaped_prev_subgoal_position = self.prev_subgoal_position.reshape(1, -1)
            reshaped_subgoal_position = self.subgoal_position.reshape(1, -1)
            #self.low_con.curr_train_metrics['subgoals_direction_diff'] = torch.nn.CosineSimilarity(reshaped_subgoal_position, reshaped_prev_subgoal_position)[0][0]
            
            subgoals_direction_diff = torch.nn.CosineSimilarity(reshaped_subgoal_position,
            reshaped_prev_subgoal_position)[0][0]
        
        return state_reached_diff, state_reached_direction_diff, subgoals_mag_diff, subgoals_direction_diff
    
    def _evaluate_high(self):
        """
        Evaluate how the current high level agent
        is doing using a perfect low level controller.
        (This has limited validity since it hasn't been trained
        with a perfect subagent)
        """
        # sample environment states that satisfy the
        # sub-agent cost function
        raise NotImplementedError("Method is not yet implemented")

    def subgoal_transition(self, s, sg, n_s):
        """
        Subgoal transition function, provided as input to the low
        level controller.
        params:
            s: state (torch.Tensor)
            sg: subgoal (torch.Tensor)
            n_s: next state (torch.Tensor)
        return:
            tr_sg: transitioned subgoal
        """
        return self.state_compressor.eval(s) + sg - self.state_compressor.eval(n_s)
    
    def low_reward(self, s, sg, n_s):
        """
        Reward function for low level controller.
        rewards the low level controller for getting close to the
        subgoals assigned to it.
        params:
            s: state (torch.Tensor)
            sg: subgoal (torch.Tensor)
            n_s: next state (torch.Tensor)
        return:
            rew: reward (float)
        """
        abs_s =  self.state_compressor.eval(s) + sg
       
        rew = -np.linalg.norm(abs_s - self.state_compressor.eval(n_s), 2)        
        #rew = -np.sqrt(np.sum((abs_s - self.state_compressor.eval(n_s))**2))
        return rew
    
    def end_step(self):
        """
        End the current step.
        params:
            None
        return:
            None
        """ 
        self.episode_subreward += self.sr
        self.sg = self.n_sg
    
    def end_episode(self, episode, logger=None):
        """
        End the current episode.
        params:
            episode: current episode (int)
            logger:
        return:
            None
        """
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
        """
        Save the agent's networks.
        params:
            episode: current episode (int)
        return:
            None
        """
        self.low_con.save(episode)
        self.high_con.save(episode)
    
    def load(self, episode):
        """
        Load the agent's networks
        params:
            episode: episode number to load.
        return:
            None
        """
        self.low_con.load(episode)
        self.high_con.load(episode)


class BaymaxAgent(HiroAgent):
    """
    BaymaxAgent is a HRL Agent that learns an internal goal space for the Subagent.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize BaymaxAgent.
        """
        # gather BaymaxAgent arguments
        state_compr_type = kwargs.pop('type_sc')
        state_compr_batch_size = kwargs.pop('batch_size_sc')
        state_compr_lr = kwargs.pop('lr_sc')
        state_compr_weight_decay = 0
        state_compr_temperature = kwargs.pop('temp_sc')

        self.state_compr_time_horizon = kwargs.pop('time_horizon_sc')
        self.state_compr_train_freq = kwargs.pop('train_freq_sc')
        self.state_compr_type_is_enc = True if state_compr_type == 'enc' else False
        
        # initialize superclass
        super(BaymaxAgent, self).__init__(*args, **kwargs)
        
        if self.state_compr_type_is_enc:
            state_compr_network = EncoderNetwork(state_dim=kwargs['state_dim'], subgoal_dim=kwargs['subgoal_dim']).to(device)
            loss_fn = SimCLR_TT_Loss(cosine_sim, batch_size=state_compr_batch_size, temperature=state_compr_temperature)
            # overwrite compressor property
            self.state_compressor = EncoderCompressor(
                model_path=kwargs['model_path'],
                network=state_compr_network,
                loss_fn=loss_fn,
                learning_rate=state_compr_lr,
                weight_decay=state_compr_weight_decay,
                )
        else:
            state_compr_network = AutoEncoderNetwork(state_dim=kwargs['state_dim'], subgoal_dim=kwargs['subgoal_dim']).to(device)
            loss_fn = torch.nn.MSELoss()
            # overwrite compressor property
            self.state_compressor = AutoEncoderCompressor(
                model_path=kwargs['model_path'],
                network=state_compr_network,
                loss_fn=loss_fn,
                learning_rate=state_compr_lr,
                weight_decay=state_compr_weight_decay,
                )
        
        # append compressor to controllers list for easier logging
        # even though it is not really a compressor
        self.controllers.append(self.state_compressor)
        
        # modify the subgoal limits to be at -0.25, 0.25 to reflect the
        # encoder properties
        self.subgoal = Subgoal(kwargs['subgoal_dim'], limits = np.ones(kwargs['state_dim'], dtype=np.float32)*(-0.25))
        self.sg = self.subgoal.action_space.sample()

    def save(self, episode):
        """
        Save the agent's networks.
        params:
            episode: current episode (int)
        return:
            None
        """
        self.low_con.save(episode)
        self.high_con.save(episode)
        self.state_compressor.save(episode)
    
    def load(self, episode):
        """
        Load the agent's networks
        params:
            episode: episode number to load.
        return:
            None
        """
        self.low_con.load(episode)
        self.high_con.load(episode)
        self.state_compressor.load(episode)


    def train(self, global_step):
        """
        Train the agent's controllers.
        params:
            global_step: current time step (int)
        return:
            losses: Dictionary of current losses (dict)
            td_errors: Dictionary of current TD Errors (dict)
        """
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
            
            if global_step % self.state_compr_train_freq == 0:
                # train the state compressor
                compressor_loss = self.state_compressor.train(self.replay_buffer_meta, time_horizon=self.state_compr_time_horizon)
                losses.update({'compressor_loss': compressor_loss})
                
        
        return losses, td_errors
    
    # TODO: implement an additional reward function that is based on the recent danijar hafner paper

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
