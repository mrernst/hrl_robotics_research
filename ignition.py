#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----

from itertools import product
from util.launcher import Launcher


from absl import flags, app
from ml_collections.config_flags import config_flags

# ----------------
# main program
# ----------------
def main(_argv):
    print(FLAGS.config)
    LOCAL = False
    TEST = False
    USE_CUDA = True
    
    JOBLIB_PARALLEL_JOBS = 2  # or os.cpu_count() to use all cores
    N_SEEDS = 1
    
    launcher = Launcher(exp_name='xx1',
                        python_file='main',
                        project_name='luna',
                        base_dir='./save/',
                        n_exps=N_SEEDS,
                        joblib_n_jobs=JOBLIB_PARALLEL_JOBS,
                        n_cores=JOBLIB_PARALLEL_JOBS * 1,
                        memory=5000,
                        days=3,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        partition='sleuths',
                        # conda_env='base',
                        # reservation='triesch-shared',
                        # gres='gpu:rtx2080ti:1' if USE_CUDA else None,
                        gres='gpu:rtx2070super:1' if USE_CUDA else None,
                        mail='mernst@fias.uni-frankfurt.de',
                        use_timestamp=True,
                        use_underscore_argparse=True
                        )
    
    
    # actor_learning_rates = [0.0001, 0.001, 0.01]
    # critic_learning_rates = [0.0001, 0.001, 0.01]
    # # #b_c_list = [11, 12]
    # # #boolean_list = [True, False]
    # # 
    # # #launcher.add_default_params(default='b')
    # # 
    # for ac_lr, cr_lr in product(actor_learning_rates, critic_learning_rates):
    #     # get the default parameters agent and main
    #     # add experiments directly to the config
    #     # b/c joblib passes arguments to experiment
    #     if LOCAL:
    #         FLAGS.config.agent.sub.actor_lr = ac_lr
    #         FLAGS.config.agent.sub.critic_lr = cr_lr
    #         launcher.add_experiment(**FLAGS.config)
    #     else:
    #         launcher.add_experiment(**{
    #             'config.main.env_name': 'PointMaze1-v0',
    #             'config.main.max_timesteps': 5e6,
    #             'config.agent.sub.actor_lr': ac_lr,
    #             'config.agent.sub.critic_lr': cr_lr,
    #             'config.agent.agent_type': 'hiro',
    #             'config.agent.subgoal_dim': 3,
    #         })
    # 
    agent_type = ['flat', 'hiro']
    for at in agent_type:
        if LOCAL:
            pass
        else:
            launcher.add_experiment(**{
                'config.agent.agent_type': at,
                'config.main.env_name': 'Antmaze-v1',
                'config.main.max_timesteps': 5e6,
                # 'config.agent.subgoal_dim': 3,
            })
        
    # launcher.add_experiment(**{
    #     'config.agent.agent_type': 'hiro', 
    #     'config.main.max_timesteps': 5e6,
    #     'config.agent.subgoal_dim': 15,
    #     })
    # 
    launcher.run(LOCAL, TEST)
    


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file('config', default='./config/default.py')
    app.run(main)
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
