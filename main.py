#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import argparse
import os
import sys
import time
import copy

import numpy as np
import torch
import gym

from absl import flags, app
from ml_collections.config_flags import config_flags

from util.launcher import get_default_params, run_experiment, \
    add_launcher_base_args, save_args
from util.replay_buffer import ReplayBuffer
import algo.td3 as TD3
from util.logger import MetricLogger, log_tensor_stats
from util.utils import get_obs_array

from agent.flat import FlatAgent
from agent.hierarchical import HiroAgent

# custom functions
# -----

def final_evaluation(main_cnf, timestep, env, agent, results_dir):

    #rewards, success_rate = agent.evaluate_policy(env, main_cnf.eval_episodes, main_cnf.render, main_cnf.save_video, main_cnf.sleep)
    rewards, success_rate = agent.evaluate_policy(env, 10, True, True, -1, results_dir, timestep)
    
    print('mean:{mean:.2f}, \
            std:{std:.2f}, \
            median:{median:.2f}, \
            success:{success:.2f}'.format(
                mean=np.mean(rewards), 
                std=np.std(rewards), 
                median=np.median(rewards), 
                success=success_rate))

def evaluate(timestep, env, agent):
    rewards, success_rate = agent.evaluate_policy(env)
    #self.logger.write('Success Rate', success_rate, e)
    
    print(" " * 80 + "\r" + "---------------------------------------")
    print('Total T: {timestep}, Reward - mean: {mean:.2f}, std: {std:.2f}, median: {median:.2f}, success:{success:.2f}'.format(
        timestep=timestep+1, 
        mean=np.mean(rewards), 
        std=np.std(rewards), 
        median=np.median(rewards), 
        success=success_rate))
    print("---------------------------------------")
    
    return np.mean(rewards)


def experiment(main, agent, seed, results_dir, **kwargs):
    """
    wrapper function to translate variable names and to catch
    unneeded additional joblib variables in local execution
    """
    main_cnf = main
    agent_cnf = agent
    results_dir = str(results_dir).replace('\n ', '/').replace(': ','_')
    # create filename and results directory
    os.makedirs(results_dir, exist_ok=True)
    # save arguments
    # TODO make this work with nested dicts
    #save_args(results_dir, locals(), git_repo_path='./')
    
    file_name = f"{agent_cnf.algorithm_name}_{main_cnf.env_name}_{seed}"
    
    print("---------------------------------------")
    print(
        f"Policy: {agent_cnf.algorithm_name}, Env: {main_cnf.env_name}, Seed: {seed}")
    print("---------------------------------------")
    
    # create world
    import env.mujoco as emj
    env = gym.make(main_cnf.env_name)

    #EnvWithGoal(create_maze_env(main_cnf.env_name), main_cnf.env_name)
    
    # set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    results_dir = os.path.join(results_dir, str(seed))
    
    # environment parameters
    state_dim =  env.observation_space['observation'].shape[0]
    action_dim =  env.action_space.shape[0]
    max_action =  torch.Tensor(env.action_space.high * np.ones(action_dim))
    goal_dim = env.observation_space['desired_goal'].shape[0]
    
    # spawn agents
    if agent_cnf.agent_type == 'flat':
        agent = FlatAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            max_action=max_action,
            model_save_freq=main_cnf.model_save_freq,
            model_path=f'{results_dir}/{file_name}',
            buffer_size=agent_cnf.sub.buffer_size,
            batch_size=agent_cnf.sub.batch_size,
            start_timesteps=main_cnf.start_timesteps
            )
    elif agent_cnf.agent_type == 'hiro':
        agent = HiroAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            subgoal_dim=main_cnf.subgoal_dim,
            max_action_low=max_action,
            start_timesteps=main_cnf.start_timesteps,
            model_path=f'{results_dir}/{file_name}',
            model_save_freq=main_cnf.model_save_freq,
            buffer_size_high=agent_cnf.meta.buffer_size,
            buffer_size_low=agent_cnf.sub.buffer_size,
            batch_size_high=agent_cnf.meta.batch_size,
            batch_size_low=agent_cnf.sub.batch_size,
            buffer_freq=agent_cnf.meta.buffer_freq,
            train_freq=agent_cnf.meta.train_freq,
            reward_scaling=agent_cnf.meta.reward_scaling,
            policy_freq_high=agent_cnf.meta.policy_freq,
            policy_freq_low=agent_cnf.sub.policy_freq,
            )
    else:
        raise NotImplementedError('Choose between flat and hierarchical (hiro) agent')
    
    if main_cnf.train:
        training_loop(
            main_cnf = main_cnf,
            agent_cnf = agent_cnf,
            env = env,
            agent = agent,
            seed = seed,
            results_dir = results_dir
        )
    if main_cnf.evaluate:
        if main_cnf.load_model:
            # load agent from file...
            print(" " * 80 + "\r" + "---------------------------------------")
            print(
              f"[INFO] Attempting loading model at episode {main_cnf.load_episode}]",
              end="\r")
        
            agent.load(main_cnf.load_episode)
            print("---------------------------------------")
        final_evaluation(main_cnf, main_cnf.max_timesteps, env, agent, results_dir)
    
     
    # space for post experiment analysis
    
    pass


def training_loop(
    main_cnf,  # main experiment configuration
    agent_cnf,  # agent configuration
    env, # environment
    agent, # agent to act
    seed=0,  # This argument is mandatory
    results_dir='./save'  # This argument is mandatory
):   
    """
    main experiment loop for training the RL Agent
    """

    # reset environment
    obs, done = env.reset(), False
    fg = obs['desired_goal']
    s = obs['observation']
    agent.set_final_goal(fg)    


    # start logging of parameters
    logger = MetricLogger(results_dir)

    # training loop
    for t in range(int(main_cnf.max_timesteps)):
        
        # Choose and take action
        a, r, n_s, done = agent.step(s, env, logger.curr_ep_steps , t, explore=True)
        
        done_bool = float(
            done) if logger.curr_ep_steps < env._max_episode_steps else 0

        # remember (store in buffer)
        agent.append(logger.curr_ep_steps, s, a, n_s, r, done)
        
        # train
        metrics = agent.train(t)
        
        # logging / updating metrics
        logger.log_step(r, None, None)
        
        # update state
        s = n_s
        agent.end_step()


        # episode is done
        if done:
            agent.end_episode(logger.episode_number, logger.writer)
            # +1 to account for 0 indexing. +0 on ep_timesteps
            # since it will increment +1 even if done=True
            if main_cnf.verbose:
                print(" " * 80 + "\r" +
                      f"Total T: {t+1} Episode Num: {logger.episode_number+1} \
                    Episode T: {logger.curr_ep_steps} Reward: {logger.curr_ep_reward:.3f}",
                      end="\r")
            

            # Reset environment
            obs, done = env.reset(), False
            fg = obs['desired_goal']
            s = obs['observation']
            agent.set_final_goal(fg) 
            
            # episode based metrics, automatically resets episode timesteps
            logger.log_episode()

        # Evaluate episode
        if (t + 1) % main_cnf.eval_freq == 0:
            mean_eval_reward = evaluate(t, env, agent)
            # log results
            logger.write_to_tensorboard(t)
            logger.writer.add_scalar(
                'training/eval_reward', mean_eval_reward, t)
            for con in agent.controllers:
                for k in con.curr_train_metrics.keys():
                    logger.writer.add_scalar(
                        f"agent/{k}", con.curr_train_metrics[k], t)
                log_tensor_stats(torch.cat([p.flatten() for p in con.actor.parameters(
                )]).detach(), "agent/actor/weights", logger.writer, t)
                log_tensor_stats(torch.cat([p.flatten() for p in con.critic.parameters(
                )]).detach(), "agent/critic/weights", logger.writer, t)
                


def main(_argv):
    print(FLAGS.config)
    run_experiment(experiment, FLAGS.config)
# ----------------
# main program
# ----------------


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
