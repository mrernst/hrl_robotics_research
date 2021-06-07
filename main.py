# Python
import numpy as np
import os
import random, datetime
import argparse

# Plotting
import matplotlib
import matplotlib.pyplot as plt

import imageio

# Neural Networks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

# Gym environment
import pybullet as p
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
from gym import spaces

# Custom utilities
from util.agent import *
from util.metrics import MetricLogger, TBMetricLogger


# Flags
parser = argparse.ArgumentParser()
parser.add_argument(
	 "-t",
	 "--testrun",
	 default=False,
	 dest='testrun',
	 action='store_true',
	 help='reduced configuration for debugging')
parser.add_argument(
	 "-v",
	 "--verbose",
	 default=False,
	 dest='verbose',
	 action='store_true',
	 help='print verbose error messages')
parser.add_argument(
	 "-e",
	 "--evaluate",
	 default=False,
	 dest='evaluate',
	 action='store_true',
	 help='evaluate trained agent')
parser.add_argument(
	 "-a",
	 "--algorithm",
	 type=str,
	 default='dqn',
	 help="type of algorithm, choose 'dqn' or 'ppo'")

args = parser.parse_args()

# set global device variable, gpu vs cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# helper functions to export movies
def embed_mp4(filename):
	import base64
	import IPython
	"""Embeds an mp4 file in the notebook."""
	video = open(filename,'rb').read()
	b64 = base64.b64encode(video)
	tag = '''
	<video width="640" height="480" controls>
	<source src="data:video/mp4;base64,{0}" type="video/mp4">
	Your browser does not support the video tag.
	</video>'''.format(b64.decode())
	
	return IPython.display.HTML(tag)


def create_policy_eval_video(policy, filename, num_episodes=5, fps=60):
	filename = filename + ".mp4"
	with imageio.get_writer(filename, fps=fps) as video:
		for _ in range(num_episodes):
			env.reset()
			state = get_screen()
			stacked_states = collections.deque(STACK_SIZE*[state],maxlen=STACK_SIZE)
			done = False
			video.append_data(env.render())
			#video.append_data(state[0,0].numpy())
			while not done:
				stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
				# Select and perform an action
				action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
				_, reward, done, _ = env.step(action.item())
				# Observe new state
				next_state = get_screen()
				#print(next_state[0,0].numpy().type)
				stacked_states.append(next_state)
				video.append_data(env.render())
				#video.append_data(next_state[0,0].numpy())
	return embed_mp4(filename)


# helper function to extract the screen from the environment
def get_screen(preprocess):
	global stacked_screens
	screen = env._get_observation().transpose((2, 0, 1))
	
	screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
	screen = torch.from_numpy(screen)
	# Resize, and add a batch dimension (BCHW)
	return preprocess(screen).unsqueeze(0).to(device)

# main program
if __name__ == '__main__':
	
	# load and reset environment
	env = KukaDiverseObjectEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20)
	env.cid = p.connect(p.DIRECT)
	env.reset()
	
	# define image preprocessing
	preprocess = T.Compose([T.ToPILImage(),
		T.Grayscale(num_output_channels=1),
		T.Resize(40, interpolation=Image.CUBIC),
		T.ToTensor()])
	
	# setup save directory
	save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
	save_dir.mkdir(parents=True)
	
	checkpoint = None if not(args.evaluate) else None
	
	# choose algorithm/agent
	if args.algorithm == 'dqn':
		action_space_n = env.action_space.n	
		agent = AgentQ(state_dim=10, action_dim=action_space_n, save_dir=save_dir, checkpoint=checkpoint)
	elif args.algorithm == 'ppo':
		raise NotImplementedError('{} is not yet implemented'.format(args.algorithm))
	else:
		raise NotImplementedError('{} is not yet implemented'.format(args.algorithm))
	
	
	logger = MetricLogger(save_dir)
	tblogger = TBMetricLogger(save_dir)
	episodes = 10000000
	
	
	if __name__ == '__main__':
		pass
	
	for e in range(episodes):
	
		state = env.reset()
	
		while True:
	
			# 3. Render environment (the visual) [WIP]
			# env.render()
			
			# 4. Run agent on the state
			action = agent.act(state)
			#action = env.action_space.sample()
	
			# 5. Agent performs action
			next_state, reward, done, info = env.step(action)
			
			if args.verbose:
				print("Action Taken  ",action)
				print("Observation   ",next_state)
				print("Reward Gained ",reward)
				print("Info          ",info,end='\n\n')
			
			# 6. Remember
			agent.add_to_memory(state, next_state, action, reward, done)
	
			# 7. Learn
			q, loss = agent.learn()
	
			# 8. Logging
			logger.log_step(reward, loss, q)
			tblogger.log_step(reward, loss, q, success=env.landed_ticks)
	
			# 9. Update state
			state = next_state
	
			# 10. Check if end of game
			if done:
				break
	
		logger.log_episode()
		tblogger.log_episode()
		
		if e % 20 == 0:
			logger.record(
				episode=e,
				epsilon=agent.exploration_rate,
				step=agent.curr_step
			)
	
	tblogger.writer.close()
	
	# final policy evaluation
	create_policy_eval_video(policy_net, "final_performance", num_episodes=10, fps=10)