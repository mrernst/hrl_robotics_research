#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----

import inspect
import json
import os
from copy import copy

import git
import numpy as np
from joblib import Parallel, delayed
import datetime
from importlib import import_module
from itertools import product

# custom classes
# -----

class Launcher(object):
    """
    Creates and starts jobs with Joblib or SLURM.

    """

    def __init__(self, exp_name, python_file, n_exps, n_cores=1, memory=2000,
                 days=0, hours=24, minutes=0, seconds=0,
                 project_name=None, base_dir=None, joblib_n_jobs=None, conda_env=None, gres=None, partition=None, begin=None,
                 use_timestamp=False, use_underscore_argparse=False, max_seeds=10000):
        """
        Constructor.

        Args:
            exp_name (str): name of the experiment
            python_file (str): prefix of the python file that runs a single experiment
            n_exps (int): number of experiments
            n_cores (int): number of cpu cores
            memory (int): maximum memory (slurm will kill the job if this is reached)
            days (int): number of days the experiment can last (in slurm)
            hours (int): number of hours the experiment can last (in slurm)
            minutes (int): number of minutes the experiment can last (in slurm)
            seconds (int): number of seconds the experiment can last (in slurm)
            project_name (str): name of the project for slurm. This is important if you have
                different projects (e.g. in the hhlr cluster)
            base_dir (str): path to directory to save the results (in hhlr results are saved to /work/scratch/$USER)
            joblib_n_jobs (int or None): number of parallel jobs in Joblib
            conda_env (str): name of the conda environment to run the experiments in
            gres (str): request cluster resources. E.g. to add a GPU in the IAS cluster specify gres='gpu:rtx2080:1'
            partition (str): the partition to use in case of slurm execution. If None, no partition is specified.
            begin (str): start the slurm experiment at a given time (see --begin in slurm docs)
            use_timestamp (bool): add a timestamp to the experiment name
            use_underscore_argparse (bool): whether to use underscore '_' in argparse instead of dash '-'
            max_seeds (int): interval [1, max_seeds-1] of random seeds to sample from

        """
        self._exp_name = exp_name
        self._python_file = python_file
        self._n_exps = n_exps
        self._n_cores = n_cores
        self._memory = memory
        self._duration = Launcher._to_duration(days, hours, minutes, seconds)
        self._project_name = project_name
        assert (joblib_n_jobs is None or joblib_n_jobs >
                0), "joblib_n_jobs must be None or > 0"
        self._joblib_n_jobs = joblib_n_jobs
        self._conda_env = conda_env
        self._gres = gres
        self._partition = partition
        self._begin = begin

        self._experiment_list = list()
        self._default_params = dict()

        if use_timestamp:
            self._exp_name += datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')

        base_dir = './logs' if base_dir is None else base_dir
        self._exp_dir_local = os.path.join(base_dir, self._exp_name)

        scratch_dir = os.path.join('/work', 'scratch', os.getenv('USER'))
        if os.path.isdir(scratch_dir):
            self._exp_dir_slurm = os.path.join(scratch_dir, self._exp_name)
        else:
            self._exp_dir_slurm = self._exp_dir_local

        self._use_underscore_argparse = use_underscore_argparse

        if n_exps >= max_seeds:
            max_seeds = n_exps + 1
            print(
                f"max_seeds must be larger than the number of experiments. Setting max_seeds to {max_seeds}")
        self._max_seeds = max_seeds

    def add_experiment(self, **kwargs):
        self._experiment_list.append(kwargs)

    def add_default_params(self, **kwargs):
        self._default_params.update(kwargs)

    def run(self, local, test=False):
        if local:
            self._run_joblib(test)
        else:
            self._run_slurm(test)

        self._experiment_list = list()

    def generate_slurm(self):
        project_name_option = ''
        partition_option = ''
        begin_option = ''
        gres_option = ''

        if self._project_name:
            project_name_option = '#SBATCH -A ' + self._project_name + '\n'
        if self._partition:
            partition_option += f'#SBATCH -p {self._partition}\n'
        if self._begin:
            begin_option += f'#SBATCH --begin={self._begin}\n'
        if self._gres:
            print(self._gres)
            gres_option += '#SBATCH --gres=' + str(self._gres) + '\n'

        joblib_seed = ''
        if self._joblib_n_jobs is not None:
            joblib_seed = f"""\
# Joblib seed
aux=$(( $SLURM_ARRAY_TASK_ID + 1 ))
aux=$(( $aux * $3 ))
if (( $aux <= $2 ))
then
  JOBLIB_SEEDS=$3
else
  JOBLIB_SEEDS=$(( $2 % $3 ))
fi
"""
        execution_code = ''
        if self._conda_env:
            if os.path.exists('/home/{}/miniconda3'.format(os.getenv('USER'))):
                execution_code += f'eval \"$(/home/{os.getenv("USER")}/miniconda3/bin/conda shell.bash hook)\"\n'
            elif os.path.exists(f'/home/{os.getenv("USER")}/anaconda3'):
                execution_code += f'eval \"$(/home/{os.getenv("USER")}/anaconda/bin/conda shell.bash hook)\"\n'
            else:
                raise Exception(
                    'You do not have a /home/USER/miniconda3 or /home/USER/anaconda3 directories')
            execution_code += f'conda activate {self._conda_env}\n\n'
            execution_code += f'python {self._python_file}.py \\'
        else:
            execution_code += f'python3  {self._python_file}.py \\'

        experiment_args = '\t\t'
        if self._joblib_n_jobs is not None:
            experiment_args += r'${@:4}'
        else:
            experiment_args += r'${@:2}'
        experiment_args += ' \\'

        if self._use_underscore_argparse:
            result_dir_code = '\t\t--results_dir $1'
        else:
            result_dir_code = '\t\t--results-dir $1'

        joblib_code = ''
        if self._joblib_n_jobs is not None:
            joblib_code = f'\\\n\t\t--joblib-n-jobs $3  '
            N_EXPS = self._n_exps
            N_JOBS = self._joblib_n_jobs
            if N_EXPS < N_JOBS:
                joblib_code += f'--joblib-n-seeds $2 \n'
            elif N_EXPS % N_JOBS == 0:
                joblib_code += f'--joblib-n-seeds $3 \n'
            elif N_EXPS % N_JOBS != 0:
                joblib_code += '--joblib-n-seeds ${JOBLIB_SEEDS} \n'
            else:
                raise NotImplementedError

        code = f"""\
#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

# Optional parameters
{project_name_option}{partition_option}{begin_option}{gres_option}
# Mandatory parameters
#SBATCH -J {self._exp_name}
#SBATCH -a 0-{self._n_exps - 1 if self._joblib_n_jobs is None else self._n_exps // self._joblib_n_jobs}
#SBATCH -t {self._duration}
#SBATCH -n 1
#SBATCH -c {self._n_cores}
#SBATCH --mem-per-cpu={self._memory}
#SBATCH -o {self._exp_dir_slurm}/%A_%a.out
#SBATCH -e {self._exp_dir_slurm}/%A_%a.err

###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

{joblib_seed}
# Program specific arguments
{execution_code}
{experiment_args}
\t\t--seed $SLURM_ARRAY_TASK_ID \\
{result_dir_code} {joblib_code}
"""
        return code

    def save_slurm(self):
        code = self.generate_slurm()

        os.makedirs(self._exp_dir_slurm, exist_ok=True)
        script_name = "slurm_" + self._exp_name + ".sh"
        full_path = os.path.join(self._exp_dir_slurm, script_name)

        with open(full_path, "w") as file:
            file.write(code)

        return full_path

    def _run_slurm(self, test):
        full_path = self.save_slurm()

        for exp in self._experiment_list:
            command_line_arguments = self._convert_to_command_line(
                exp, self._use_underscore_argparse)
            if self._default_params:
                command_line_arguments += ' '
                command_line_arguments += self._convert_to_command_line(self._default_params,
                                                                        self._use_underscore_argparse)
            results_dir = self._generate_results_dir(self._exp_dir_slurm, exp)

            command = "sbatch " + full_path + ' ' + results_dir
            if self._joblib_n_jobs is not None:
                command += ' ' + str(self._n_exps) + ' ' + \
                    str(self._joblib_n_jobs)
            command += ' ' + command_line_arguments

            if test:
                print(command)
            else:
                os.system(command)

    def _run_joblib(self, test):
        if not test:
            os.makedirs(self._exp_dir_local, exist_ok=True)

        module = import_module(self._python_file)
        experiment = module.experiment

        if test:
            if self._default_params:
                default_params = str(self._default_params).replace('{', ', ').replace('}', ', ') \
                    .replace(': ', '=').replace('\'', '')
            else:
                default_params = ''

            for exp, i in product(self._experiment_list, range(self._n_exps)):
                results_dir = self._generate_results_dir(
                    self._exp_dir_local, exp)
                params = str(exp).replace(
                    '{', '(').replace('}', '').replace(': ', '=').replace('\'', '')
                print('experiment' + params + default_params + 'seed=' +
                      str(i) + ', results_dir=' + results_dir + ')')
        else:
            params_dict = get_default_params(experiment)
            
            Parallel(n_jobs=self._joblib_n_jobs)(delayed(experiment)(**params)
                                                 for params in self._generate_exp_params(params_dict))

    @staticmethod
    def _generate_results_dir(results_dir, exp):
        for key, value in exp.items():
            subfolder = key + '_' + str(value)
            results_dir = os.path.join(results_dir, subfolder)

        return results_dir

    def _generate_exp_params(self, params_dict):
        params_dict.update(self._default_params)

        seeds = np.arange(self._n_exps)
        for exp, seed in product(self._experiment_list, seeds):
            params_dict.update(exp)
            params_dict['seed'] = int(seed)
            params_dict['results_dir'] = self._generate_results_dir(
                self._exp_dir_local, exp)
            yield params_dict

    @staticmethod
    def _convert_to_command_line(exp, use_underscore_argparse):
        command_line = ''
        for key, value in exp.items():
            if use_underscore_argparse:
                new_command = '--' + key + ' '
            else:
                new_command = '--' + key.replace('_', '-') + ' '

            if isinstance(value, bool):
                new_command = new_command if value else ''
            else:
                new_command += str(value) + ' '

            command_line += new_command

        return command_line

    @staticmethod
    def _to_duration(days, hours, minutes, seconds):
        h = "0" + str(hours) if hours < 10 else str(hours)
        m = "0" + str(minutes) if minutes < 10 else str(minutes)
        s = "0" + str(seconds) if seconds < 10 else str(seconds)

        return str(days) + '-' + h + ":" + m + ":" + s


def get_default_params(func):
    signature = inspect.signature(func)
    defaults = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            defaults[k] = v.default

    return defaults


def run_experiment(func, args):
    joblib_n_jobs = copy(args['joblib_n_jobs'])
    joblib_n_seeds = copy(args['joblib_n_seeds'])
    initial_seed = copy(args['seed'])
    if joblib_n_jobs is not None:
        initial_seed *= joblib_n_jobs
    try:
      args.unlock()
    except:
      pass
    del args['joblib_n_jobs']
    del args['joblib_n_seeds']
    try:
      args.lock()
    except:
      pass

    def generate_joblib_seeds(params_dict):
        final_seed = initial_seed + 1 if joblib_n_seeds is None else joblib_n_seeds
        seeds = np.arange(initial_seed, final_seed, dtype=int)
        for seed in seeds:
            params_dict['seed'] = int(seed)
            yield params_dict

    Parallel(n_jobs=joblib_n_jobs)(delayed(func)(**params)
                                   for params in generate_joblib_seeds(args))


def add_launcher_base_args(parser):
    arg_default = parser.add_argument_group('Default')
    arg_default.add_argument('--seed', type=int)
    arg_default.add_argument('--results_dir', type=str)
    arg_default.add_argument('--joblib-n-jobs', type=int)
    arg_default.add_argument('--joblib-n-seeds', type=int)
    return parser


def save_args(results_dir, args, git_repo_path=None, seed=None):
    repo = git.Repo(git_repo_path, search_parent_directories=True)
    args['git_hash'] = repo.head.object.hexsha
    args['git_url'] = repo.remotes.origin.url

    filename = 'args.json' if seed is None else f'args-{seed}.json'
    # Save args
    l = []
    for a in args:
      try:
        l.append(args[a].items())
      except:
        l.append((a,args[a]))
    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump(l, f, indent=2)

    del args['git_hash']
    del args['git_url']


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
