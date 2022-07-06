# ----------------
# import libraries
# ----------------


# standard libraries
# -----

import ml_collections

# custom functions
# -----

# --------------------------
# main experiment parameters
# --------------------------

def get_config():
	"""
	Get main parameters.
	For each experiment, change these parameters manually for different
	experiments.
	"""
	cnf = ml_collections.ConfigDict(
		{
			'main': {
				# OpenAI gym environment name
				'env_name': "AntMaze1-v0",
				# General Training parameter
				'train': True,
				# General Evaluation parameter
				'evaluate': True,
				# Time steps initial random policy is used
				'start_timesteps': 25e3,
				# How often (time steps) we evaluate
				'eval_freq': 1e4,
				# How often (time steps) we save an eval. video
				'eval_video_freq': 1e5,
				# Max time steps to run environment
				'max_timesteps': 10e6,
				# Save model and optimizer parameters
				'save_model': True,
				# Save model every n episodes 
				# TODO: should be n training steps
				'model_save_freq': 5000,
				# Load model from file
				'load_model': False,
				# Which episode save should be loaded (-1 == last avail.)
				'load_episode': -1,
				# Print information to CLI
				'verbose': False,
			},
			'agent': {
				# Agent name (flat or hiro)
				'agent_type': "flat",
				# Algorithm name (TD3, or else if implemented)
				'algorithm_name': "TD3",
				# Subgoal dimension
				'subgoal_dim': 15,
				# sub agent configuration
				'sub': {
					# Std of Gaussian exploration noise
					'expl_noise': 1., #sub agent has +-16 actionspace
					# Prioritized Experience Replay (0 = regular buffers)
					'prio_exp_replay': 0,
					# Buffer size
					'buffer_size': int(2**15),#int(2e5),
					# Batch size for both actor and critic
					'batch_size': 256,
					# Discount factor
					'discount': 0.99,
					# Target network update rate
					'tau': 0.005,
					# Noise added to target policy during critic update
					'policy_noise': 0.2,
					# Range to clip target policy noise
					'noise_clip': 0.5,
					# Frequency of delayed policy updates
					'policy_freq': 2,
					# Learning rate of the actor
					'actor_lr': 0.0001, #0.0001
					# Learning rate of the critic
					'critic_lr': 0.001, #0.001
					# Hidden layer size of the actor
					'actor_hidden_layers': [300, 300],
					# Hidden layer size of the critic
					'critic_hidden_layers': [300, 300],
				},
				# meta agent configuration
				'meta': {
					# Std of Gaussian exploration noise
					'expl_noise': 1., #meta action has +-10 space
					# Prioritized Experience Replay (0 = regular buffers)
					'prio_exp_replay': 0,
					# Buffer size
					'buffer_size': int(2**15), #int(2e5),
					# Batch size for both actor and critic
					'batch_size': 256,
					# Discount factor
					'discount': 0.99,
					# Target network update rate
					'tau': 0.005,
					# Noise added to target policy during critic update
					'policy_noise': 0.2,
					# Range to clip target policy noise
					'noise_clip': 0.5,
					# Frequency of delayed policy updates
					'policy_freq': 2,
					# Learning rate of the actor
					'actor_lr': 0.0001, #0.0001
					# Learning rate of the critic
					'critic_lr': 0.001, #0.001
					# Hidden layer size of the actor
					'actor_hidden_layers': [300, 300],
					# Hidden layer size of the critic
					'critic_hidden_layers': [300, 300],
					# Buffer frequency determines the time until next subgoal
					'buffer_freq': 10,
					# Train frequency
					'train_freq': 10,
					# Reward scaling
					'reward_scaling': 0.1,
				}
			},
		}
	)
	
	
	# add standard parameters for launcher
	cnf.seed = 0
	cnf.results_dir = './save'
	cnf.joblib_n_jobs = 1
	cnf.joblib_n_seeds = 1
	

	return cnf



# _____________________________________________________________________________

# policy_name="TD3",
# env_name="Reacher-v2",
# start_timesteps=25e3,
# eval_freq=5e3,
# max_timesteps=1e6,
# expl_noise=0.1,
# batch_size=256,
# discount=0.99,
# tau=0.005,
# policy_noise=0.2,
# noise_clip=0.5,
# policy_freq=2,
# actor_lr=0.0003,
# critic_lr=0.0003,
# save_model=False,
# load_model="",
# verbose=False,
