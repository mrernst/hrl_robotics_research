#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----

from util.utils import darken
from util.replay_buffer import ReplayBuffer
import numpy as np
import torch
import gym
import argparse
import os
import sys


import imageio
import base64
import glfw

from gym.wrappers.monitoring import video_recorder

# utilities
# -----
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class Agent(object):
    """
    Abstract Agent Base Class that all other agent classes inherit from.
    """

    def __init__(self):
        """
        Initialize Agent Object.
        """
        raise NotImplementedError

    def set_final_goal(self, fg):
        """
        Set final goal for a hierarchical agent.
        params:
            fg: final goal (torch.Tensor)
        return:
            None
        """
        self.fg = fg

    def step(self, s, env, step, global_step=0, explore=False):
        """
        Step the simulation.
        """
        raise NotImplementedError

    def append(self, step, s, a, n_s, r, d):
        """
        Append to the agent's replay buffer.
        """
        raise NotImplementedError

    def train(self, global_step):
        """
        Train the agents controllers.
        """
        raise NotImplementedError

    def end_step(self):
        """
        End the current simulation step.
        """
        raise NotImplementedError

    def end_episode(self, episode, logger=None):
        """
        End the current simulation episode.
        """
        raise NotImplementedError

    def evaluate_policy(self, env, eval_episodes=10, render=False, save_video=False, sleep=-1, results_dir='./save', timestep=-1):
        """
        Evaluate the trained policies.
        params:
            env: gym Environment (gym.env)
            eval_episodes: number of episodes to be evaluated (int)
            render: Should evaluated episodes be rendered (bool)
            save_video: Should evaluated episodes be saved to video (bool)
            sleep: how many seconds of sleep between episodes (int)
            results_dir: directory where to save videos (str)
            timestep: global time step (int)
        return:
            rew: rewards (np.array)
            srate: successrate (float)
            
        """
        if save_video:
            from OpenGL import GL
            # env = gym.wrappers.Monitor(env, directory='video',
            #						write_upon_reset=True, force=True, resume=True, mode='evaluation')
            os.makedirs(f'{results_dir}/video/', exist_ok=True)
            video = imageio.get_writer(
                f'{results_dir}/video/t{timestep}.mp4', fps=30)
            ghostimage_list = []
            ghost_every = 10
            render = False

        success = 0
        rewards = []
        env.evaluate = True
        with torch.no_grad():
            for e in range(eval_episodes):
                obs = env.reset()
                fg = obs['desired_goal']
                fg_dim = fg.shape[0]
                s = obs['observation']
                done = False
                reward_episode_sum = 0
                step = 0

                self.set_final_goal(fg)

                while not done:
                    if render:
                        if hasattr(self, 'sg') and not hasattr(self, 'state_compr_time_horizon'):
                            # if possible render subgoal
                            env.render(subgoal=self.sg+s[:self.sg.shape[0]])
                        else:
                            env.render()
                    if sleep > 0:
                        time.sleep(sleep)

                    a, r, n_s, done = self.step(s, env, step)
                    reward_episode_sum += r

                    s = n_s
                    step += 1
                    self.end_step()
                    if save_video:
                        if hasattr(self, 'sg') and not hasattr(self, 'state_compr_time_horizon'):
                            videoframe = env.render(
                                subgoal=self.sg+s[:self.sg.shape[0]], mode='rgb_array', width=720, height=480)
                            video.append_data(videoframe)
                        if step % ghost_every == 0:
                            ghostimage_list.append(
                                np.dstack([videoframe, np.ones(videoframe.shape[:2])*255]).astype(float))
                        else:
                            videoframe = env.render(
                                mode='rgb_array', width=720, height=480)
                            video.append_data(videoframe)
                else:
                    error = np.sqrt(np.sum(np.square(fg-s[:fg_dim])))
                    print(" " * 80 + "\r" +
                          '[Eval] Goal, Curr: (%02.2f, %02.2f, %02.2f, %02.2f)     Error:%.2f' % (fg[0], fg[1], s[0], s[1], error), end='\r')
                    rewards.append(reward_episode_sum)
                    success += 1 if error <= 5 else 0

                    # if save_video:
                    # 	# create ghost image
                    # 	gimg = ghostimage_list[0]
                    # 	for i in range(len(ghostimage_list)-1):
                    # 		opacity = 0.7
                    # 		gimg = darken(gimg, ghostimage_list[i+1], opacity)
                    # 	ghostimage_list = []
                    # this is not suited for every environment, distance should be adapted for success metric
                    self.end_episode(e)
                    if hasattr(env, 'viewer') and render:
                        v = env.viewer
                        #env.viewer = None
                        glfw.destroy_window(v.window)
                        #del v

        env.evaluate = False
        # the function should also return a detailed log of states, subgoals, actions, goals
        return np.array(rewards), success/eval_episodes
