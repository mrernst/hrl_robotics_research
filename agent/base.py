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


class Agent(object):
	"""
	Abstract Agent Base Class
	"""
	def __init__(self):
		pass
	
	def set_final_goal(self, fg):
		self.fg = fg
	
	def step(self, s, env, step, global_step=0, explore=False):
		raise NotImplementedError
	
	def append(self, step, s, a, n_s, r, d):
		raise NotImplementedError
	
	def train(self, global_step):
		raise NotImplementedError
	
	def end_step(self):
		raise NotImplementedError
	
	def end_episode(self, episode, logger=None):
		raise NotImplementedError
	
	def evaluate_policy(self, env, eval_episodes=10, render=False, save_video=False, sleep=-1):
		if save_video:
			from OpenGL import GL
			env = gym.wrappers.Monitor(env, directory='video',
									write_upon_reset=True, force=True, resume=True, mode='evaluation')
			render = False
	
		success = 0
		rewards = []
		env.evaluate = True
		for e in range(eval_episodes):
			obs = env.reset()
			fg = obs['desired_goal']
			s = obs['observation']
			done = False
			reward_episode_sum = 0
			step = 0
			
			self.set_final_goal(fg)
	
			while not done:
				if render:
					env.render()
				if sleep>0:
					time.sleep(sleep)
	
				a, r, n_s, done = self.step(s, env, step)
				reward_episode_sum += r
				
				s = n_s
				step += 1
				self.end_step()
			else:
				error = np.sqrt(np.sum(np.square(fg-s[:2])))
				print('Goal, Curr: (%02.2f, %02.2f, %02.2f, %02.2f)     Error:%.2f'%(fg[0], fg[1], s[0], s[1], error))
				rewards.append(reward_episode_sum)
				success += 1 if error <=5 else 0
				self.end_episode(e)
	
		env.evaluate = False
		return np.array(rewards), success/eval_episodes
