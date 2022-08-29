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
from util.utils import MakeGoalBased

from agent.flat import FlatAgent
from agent.hierarchical import HiroAgent, BaymaxAgent

# custom functions
# -----

def evaluate_extended(timestep, env, agent, results_dir, to_video=False):
    # TODO: implement a function that additionally visualizes
    # goal and state spaces and is able to render a video alongside
    # the agent acting in the environment
    pass

def evaluate(timestep, env, agent, results_dir, to_video=False):
    rewards, success_rate = agent.evaluate_policy(env, 10, to_video, to_video, -1, results_dir, timestep + 1)
    
    print(" " * 80 + "\r" + "---------------------------------------")
    print('Total T: {timestep}, Reward - mean: {mean:.2f}, std: {std:.2f}, median: {median:.2f}, success:{success:.2f}'.format(
        timestep=timestep+1, 
        mean=np.mean(rewards), 
        std=np.std(rewards), 
        median=np.median(rewards), 
        success=success_rate))
    print("---------------------------------------")
    
    return np.mean(rewards), success_rate


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
    # save_args(results_dir, locals(), git_repo_path='./')
    save_args(results_dir, main_cnf.to_dict(), name='main_', git_repo_path='./', seed=seed)
    save_args(results_dir, agent_cnf.to_dict(), name='agent_', git_repo_path='./', seed=seed)

    file_name = f"{agent_cnf.algorithm_name}_{main_cnf.env_name}_{seed}"
    
    print("---------------------------------------")
    print(
        f"Policy: {agent_cnf.algorithm_name}, Env: {main_cnf.env_name}, Seed: {seed}")
    print("---------------------------------------")
    
    # create world
    import env.mujoco as emj
    #env = MakeGoalBased(gym.make(main_cnf.env_name))
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
    goal_dim = env.observation_space['desired_goal'].shape[0]
    action_dim =  env.action_space.shape[0]
    max_action =  torch.Tensor(env.action_space.high * np.ones(action_dim))
    
    
    # spawn agents
    if agent_cnf.agent_type == 'flat':
        agent = FlatAgent(
            state_dim=state_dim,
            goal_dim=goal_dim,
            action_dim=action_dim,
            max_action=max_action,
            model_path=f'{results_dir}/{file_name}',
            model_save_freq=main_cnf.model_save_freq,
            start_timesteps=main_cnf.start_timesteps,
            prio_exp_replay=agent_cnf.sub.prio_exp_replay,
            buffer_size=agent_cnf.sub.buffer_size,
            batch_size=agent_cnf.sub.batch_size,
            actor_lr=agent_cnf.sub.actor_lr,
            critic_lr=agent_cnf.sub.critic_lr,
            actor_hidden_layers=agent_cnf.sub.actor_hidden_layers,
            critic_hidden_layers=agent_cnf.sub.critic_hidden_layers,
            expl_noise=agent_cnf.sub.expl_noise,
            policy_noise=agent_cnf.sub.policy_noise * float(env.action_space.high[0]),
            noise_clip=agent_cnf.sub.noise_clip * float(env.action_space.high[0]),
            discount=agent_cnf.sub.discount,
            policy_freq=agent_cnf.sub.policy_freq,
            tau=agent_cnf.sub.tau,
            )
    elif agent_cnf.agent_type == 'hiro':
        agent = HiroAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            subgoal_dim=agent_cnf.subgoal_dim,
            max_action_sub=max_action,
            model_path=f'{results_dir}/{file_name}',
            model_save_freq=main_cnf.model_save_freq,
            start_timesteps=main_cnf.start_timesteps,
            # meta agent arguments
            prio_exp_replay_meta=agent_cnf.meta.prio_exp_replay,
            buffer_size_meta=agent_cnf.meta.buffer_size,
            batch_size_meta=agent_cnf.meta.batch_size,
            actor_lr_meta=agent_cnf.meta.actor_lr,
            critic_lr_meta=agent_cnf.meta.critic_lr,
            actor_hidden_layers_meta=agent_cnf.meta.actor_hidden_layers,
            critic_hidden_layers_meta=agent_cnf.meta.critic_hidden_layers,
            expl_noise_meta=agent_cnf.meta.expl_noise,
            policy_noise_meta=agent_cnf.meta.policy_noise,
            noise_clip_meta=agent_cnf.meta.noise_clip,
            discount_meta=agent_cnf.meta.discount,
            policy_freq_meta=agent_cnf.meta.policy_freq,
            tau_meta=agent_cnf.meta.tau,
            buffer_freq=agent_cnf.meta.buffer_freq,
            train_freq=agent_cnf.meta.train_freq,
            reward_scaling=agent_cnf.meta.reward_scaling,
            # sub agent arguments
            prio_exp_replay_sub=agent_cnf.sub.prio_exp_replay,
            buffer_size_sub=agent_cnf.sub.buffer_size,
            batch_size_sub=agent_cnf.sub.batch_size,
            actor_lr_sub=agent_cnf.sub.actor_lr,
            critic_lr_sub=agent_cnf.sub.critic_lr,
            actor_hidden_layers_sub=agent_cnf.sub.actor_hidden_layers,
            critic_hidden_layers_sub=agent_cnf.sub.critic_hidden_layers,
            expl_noise_sub=agent_cnf.sub.expl_noise,
            policy_noise_sub=agent_cnf.sub.policy_noise * float(env.action_space.high[0]),
            noise_clip_sub=agent_cnf.sub.noise_clip * float(env.action_space.high[0]),
            discount_sub=agent_cnf.sub.discount,
            policy_freq_sub=agent_cnf.sub.policy_freq,
            tau_sub=agent_cnf.sub.tau,
            )
    elif agent_cnf.agent_type == 'baymax':
        agent = BaymaxAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            subgoal_dim=agent_cnf.subgoal_dim,
            max_action_sub=max_action,
            model_path=f'{results_dir}/{file_name}',
            model_save_freq=main_cnf.model_save_freq,
            start_timesteps=main_cnf.start_timesteps,
            # meta agent arguments
            prio_exp_replay_meta=agent_cnf.meta.prio_exp_replay,
            buffer_size_meta=agent_cnf.meta.buffer_size,
            batch_size_meta=agent_cnf.meta.batch_size,
            actor_lr_meta=agent_cnf.meta.actor_lr,
            critic_lr_meta=agent_cnf.meta.critic_lr,
            actor_hidden_layers_meta=agent_cnf.meta.actor_hidden_layers,
            critic_hidden_layers_meta=agent_cnf.meta.critic_hidden_layers,
            expl_noise_meta=agent_cnf.meta.expl_noise,
            policy_noise_meta=agent_cnf.meta.policy_noise,
            noise_clip_meta=agent_cnf.meta.noise_clip,
            discount_meta=agent_cnf.meta.discount,
            policy_freq_meta=agent_cnf.meta.policy_freq,
            tau_meta=agent_cnf.meta.tau,
            buffer_freq=agent_cnf.meta.buffer_freq,
            train_freq=agent_cnf.meta.train_freq,
            reward_scaling=agent_cnf.meta.reward_scaling,
            # sub agent arguments
            prio_exp_replay_sub=agent_cnf.sub.prio_exp_replay,
            buffer_size_sub=agent_cnf.sub.buffer_size,
            batch_size_sub=agent_cnf.sub.batch_size,
            actor_lr_sub=agent_cnf.sub.actor_lr,
            critic_lr_sub=agent_cnf.sub.critic_lr,
            actor_hidden_layers_sub=agent_cnf.sub.actor_hidden_layers,
            critic_hidden_layers_sub=agent_cnf.sub.critic_hidden_layers,
            expl_noise_sub=agent_cnf.sub.expl_noise,
            policy_noise_sub=agent_cnf.sub.policy_noise * float(env.action_space.high[0]),
            noise_clip_sub=agent_cnf.sub.noise_clip * float(env.action_space.high[0]),
            discount_sub=agent_cnf.sub.discount,
            policy_freq_sub=agent_cnf.sub.policy_freq,
            tau_sub=agent_cnf.sub.tau,
            # state compression arguments
            type_sc=agent_cnf.compressor.type,
            batch_size_sc=agent_cnf.compressor.batch_size,
            lr_sc=agent_cnf.compressor.lr,
            temp_sc=agent_cnf.compressor.temp,
            time_horizon_sc=agent_cnf.compressor.time_horizon,
            train_freq_sc=agent_cnf.compressor.train_freq,
            )
    else:
        raise NotImplementedError('Choose between flat and hierarchical (hiro, baymax) agents')
    
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
        _, _ = evaluate(main_cnf.max_timesteps, env, agent, results_dir, to_video=True)
    
     
    # space for post experiment analysis
    # run functions of the afterburner module
    
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
    # build the corresponding test environment if available
    try:
        test_env_name = main_cnf.env_name.split('-')[0]+'Test-' + main_cnf.env_name.split('-')[1]
        test_env = gym.make(test_env_name)
    except gym.error.NameNotFound:
        print('[INFO] No test environment found, using default environment instead')
        test_env = env
    
    
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
            #print(agent.episode_subreward)
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

        # Evaluate agent
        if (t + 1) % main_cnf.eval_freq == 0 and t > main_cnf.start_timesteps:
            # log evaluation results
            
            mean_eval_reward, success_rate = evaluate(t, env, agent, results_dir)
                
            logger.writer.add_scalar(
                'evaluation/default_env/success_rate', success_rate, t)
            logger.writer.add_scalar(
                'evaluation/default_env/reward', mean_eval_reward, t)
            
            mean_eval_reward, success_rate = evaluate(t, test_env, agent, results_dir)
                
            logger.writer.add_scalar(
                'evaluation/test_env/success_rate', success_rate, t)
            logger.writer.add_scalar(
                'evaluation/test_env/reward', mean_eval_reward, t)
            # TODO: All evaluation metrics should be averages with standard deviations, such that one can better judge
            # progress
            
            
            # log general results
            
            logger.writer.add_scalar(
                'training/reward', logger.ep_rewards[-1], t)
            logger.writer.add_scalar(
                'training/episode_length', logger.ep_lengths[-1], t)
            
            for con in agent.controllers:
                for k in con.curr_train_metrics.keys():
                    logger.writer.add_scalar(
                        f"agent/{con.name}/{k}", con.curr_train_metrics[k], t)
            # check whether the controller has an actor/critic infrastructure
            if hasattr(con, 'actor'):
                log_tensor_stats(torch.cat([p.flatten() for p in con.actor.parameters(
                )]).detach(), f"agent/{con.name}/actor/weights", logger.writer, t)
                log_tensor_stats(torch.cat([p.flatten() for p in con.critic.parameters(
                )]).detach(), f"agent/{con.name}/critic/weights", logger.writer, t)
            # this also makes possible logging other objects that have a network
            # property
            else:
                log_tensor_stats(torch.cat([p.flatten() for p in con.network.parameters(
                )]).detach(), f"agent/{con.name}/network/weights", logger.writer, t)
            

        # Save a video of the evaluation
        if main_cnf.eval_video_freq > 0:
            if (t + 1) % main_cnf.eval_video_freq == 0 and t > main_cnf.start_timesteps:
                mean_eval_reward, success_rate = evaluate(t, env, agent, results_dir, to_video=True)
                
                # dev log some additional correlation images
                # TODO standardize this somehow
                if hasattr(agent.controllers[-1], "network"):
                    logger.writer.add_figure('agent/state_compressor/training/', agent.state_compressor.eval_state_info(batch_size=256, buffer=agent.replay_buffer_meta), t)
                    logger.writer.add_figure('agent/state_compressor/validation', agent.state_compressor.eval_state_info(batch_size=256), t)


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
