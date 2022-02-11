#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----

from itertools import product
from util.launcher import Launcher

# ----------------
# main program
# ----------------

if __name__ == '__main__':
    LOCAL = False
    TEST = False
    USE_CUDA = True

    JOBLIB_PARALLEL_JOBS = 4  # or os.cpu_count() to use all cores
    N_SEEDS = 1

    launcher = Launcher(exp_name='001',
                        python_file='main',
                        project_name='luna',
                        base_dir='./save/',
                        n_exps=N_SEEDS,
                        joblib_n_jobs=JOBLIB_PARALLEL_JOBS,
                        n_cores=JOBLIB_PARALLEL_JOBS * 1,
                        memory=5000,
                        days=1,
                        hours=0,
                        minutes=0,
                        seconds=0,
                        partition='sleuths',
                        # conda_env='base',
                        gres='gpu:rtx2070super:1' if USE_CUDA else None,
                        use_timestamp=True,
                        use_underscore_argparse=True
                        )

    actor_learning_rates = [0.0003, 0.003, 0.03]
    critic_learning_rates = [0.0003, 0.003, 0.03]
    #b_c_list = [11, 12]
    #boolean_list = [True, False]

    #launcher.add_default_params(default='b')

    for ac_lr, cr_lr in product(actor_learning_rates, critic_learning_rates):
        launcher.add_experiment(actor_lr=ac_lr,
                                critic_lr=cr_lr)

    launcher.run(LOCAL, TEST)

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
