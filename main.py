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

from agent.flat import Agent
# custom functions
# -----


def experiment(
    main_cnf,
    agent_cnf,
    seed=0,  # This argument is mandatory
    results_dir='./save'  # This argument is mandatory
):

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    # Save arguments
    save_args(results_dir, locals(), git_repo_path='./')

    
    file_name = f"{agent_cnf.policy_name}_{main_cnf.env_name}_{seed}"

    print("---------------------------------------")
    print(f"Policy: {agent_cnf.policy_name}, Env: {main_cnf.env_name}, Seed: {seed}")
    print("---------------------------------------")

    env = gym.make(main_cnf.env_name)
    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    # state_dim = np.sum(
    #    [env.observation_space.spaces[k].shape for k in env.observation_space.spaces.keys()])
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize RL algorithm
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": agent_cnf.discount,
        "tau": agent_cnf.tau,
    }

    if agent_cnf.policy_name == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = agent_cnf.policy_noise * max_action
        kwargs["noise_clip"] = agent_cnf.noise_clip * max_action
        kwargs["policy_freq"] = agent_cnf.policy_freq
        kwargs["actor_lr"] = agent_cnf.actor_lr
        kwargs["critic_lr"] = agent_cnf.critic_lr
        
        policy = TD3.TD3(**kwargs, name='flat')
    else:
        raise NotImplementedError()

    if main_cnf.load_model != "":
        policy_file = file_name if main_cnf.load_model == "default" else main_cnf.load_model
        policy.load(f"{results_dir}/{policy_file}")
    
    results_dir = os.path.join(results_dir, str(seed))
    
    # choose replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Initialize Agent with policy
    agent = Agent(action_dim, policy, replay_buffer, burnin=main_cnf.start_timesteps)

    # Evaluate untrained policy
    agent.eval_policy(main_cnf.env_name, seed)
    # delete: evaluations = [eval_policy(policy, main_cnf.env_name, seed)]

    # Main training loop
    # -----

    # reset environment
    state, done = env.reset(), False
    # state = np.concatenate([state[k] for k in state.keys()])

    # start logging of parameters
    # initialize logger
    logger = MetricLogger(results_dir)
    
    # training loop
    for t in range(int(main_cnf.max_timesteps)):

        # 3. Render environment [WIP]
        # env.render()

        # 4. Run agent on the state (get the action)
        # Select action randomly or according to policy

        if t < main_cnf.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, max_action, agent_cnf.expl_noise)

        # 5. Agent performs action
        next_state, reward, done, _ = env.step(action)
        # next_state = np.concatenate([next_state[k] for k in next_state.keys()])

        done_bool = float(
            done) if logger.curr_ep_steps < env._max_episode_steps else 0

        # 6. Remember
        # Store data in replay buffer
        agent.add_to_memory(state, action, next_state, reward, done_bool)

        # 9. Update state
        state = next_state
        
        # 8. Logging / updating metrics
        logger.log_step(reward, None, None)
        
        # 7. Learn
        # Train agent after collecting sufficient data
        if t >= main_cnf.start_timesteps:
            agent.learn(agent_cnf.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps
            # since it will increment +1 even if done=True
            if main_cnf.verbose:
                print(" " * 80 + "\r" +
                      f"Total T: {t+1} Episode Num: {logger.episode_number+1} \
                    Episode T: {logger.curr_ep_steps} Reward: {logger.curr_ep_reward:.3f}",
                      end="\r")

            # Reset environment
            state, done = env.reset(), False
            # state = np.concatenate([state[k] for k in state.keys()])

            # 10. Episode based metrics, automatically resets episode timesteps
            logger.log_episode()
        
        # Evaluate episode
        if (t + 1) % main_cnf.eval_freq == 0:
            eval_episodes = 10
            avg_reward = agent.eval_policy(main_cnf.env_name, seed, eval_episodes)
            if main_cnf.verbose:
                print(" " * 80 + "\r" + "---------------------------------------")
                print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
                print("---------------------------------------")
            
            # log results
            logger.write_to_tensorboard(t)
            logger.writer.add_scalar(
                'training/eval_reward', agent.evaluations[-1], t)
            for k in agent.policy.curr_train_metrics.keys():
                logger.writer.add_scalar(f"agent/{k}", agent.policy.curr_train_metrics[k], t)
            log_tensor_stats(torch.cat([p.flatten() for p in agent.policy.actor.parameters()]).detach(), "agent/actor/weights", logger.writer, t)
            log_tensor_stats(torch.cat([p.flatten() for p in agent.policy.critic.parameters()]).detach(), "agent/critic/weights", logger.writer, t)
            
            
            if main_cnf.save_model:
                policy.save(f"{results_dir}/{file_name}")
    
            # video evaluation if model is loaded
            if main_cnf.load_model != "":
                agent.create_policy_eval_video(main_cnf.env_name, seed, results_dir + f"/t_{t+1}")




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
