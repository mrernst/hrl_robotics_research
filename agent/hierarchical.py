#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----

from util.replay_buffer import TransitBuffer
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

class HIRO(object):
	"""
	"""
	def __init__(self, action_dim, policy, replay_buffer, burnin):
		pass
		
	def select_action(self, state, max_action, noise):
		pass

	def add_to_memory(self, state, action, next_state, reward, done):
		pass
		
	def learn(self, batch_size):
		pass

	def reset(self):
		pass

	def eval_policy(self, env_name, seed, eval_episodes=10):
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
