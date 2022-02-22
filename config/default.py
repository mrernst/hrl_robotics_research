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
				'env_name': "Reacher-v2",
				# Time steps initial random policy is used
				'start_timesteps': 25e3,
				# How often (time steps) we evaluate
				'eval_freq': 5e3,
				# Max time steps to run environment
				'max_timesteps': 1e6,
				# Save model and optimizer parameters
				'save_model': False,
				# Model load file name, "" doesn't load, "default" uses file_name
				'load_model': "",
				# Print information to CLI
				'verbose': False,
			},
			'agent': {
				# Policy name (TD3, or else if implemented)
				'policy_name': "TD3",
				# Std of Gaussian exploration noise
				'expl_noise': 0.1,
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
				'actor_lr': 0.0003,
				# Learning rate of the critic
				'critic_lr': 0.0003,
				# sub agent configuration
				'sub': {
					'placeholder': 1,
				},
				# meta agent configuration
				'meta': {
					'placeholder': 1,
				}
			},
		}
	)
	
	
	# add standard parameters for launcher
	cnf.seed = 0
	cnf.results_dir = './save'
	cnf.joblib_n_jobs = None
	cnf.joblib_n_seeds = None
	

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
