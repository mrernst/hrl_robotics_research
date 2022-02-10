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

from util.launcher import get_default_params, run_experiment, \
    add_launcher_base_args, save_args
from util.replay_buffer import ReplayBuffer
import algo.td3 as TD3
from util.logger import MetricLogger

from agent.agent import Agent
# custom functions
# -----


def experiment(
    policy_name="TD3",
    env_name="InvertedPendulum-v2",
    start_timesteps=25e3,
    eval_freq=5e3,
    max_timesteps=1e6,
    expl_noise=0.1,
    batch_size=256,
    discount=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_freq=2,
    save_model=False,
    load_model="",
    seed=0,  # This argument is mandatory
    results_dir='./save'  # This argument is mandatory
):

    # Create results directory
    results_dir = os.path.join(results_dir, str(seed))
    os.makedirs(results_dir, exist_ok=True)
    # Save arguments
    save_args(results_dir, locals(), git_repo_path='./')
    if save_model:
        os.makedirs(os.path.join(results_dir, '/models'), exist_ok=True)

    logger = MetricLogger(results_dir)

    file_name = f"{policy_name}_{env_name}_{seed}"
    print("---------------------------------------")
    print(f"Policy: {policy_name}, Env: {env_name}, Seed: {seed}")
    print("---------------------------------------")

    env = gym.make(env_name)
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
        "discount": discount,
        "tau": tau,
    }

    if policy_name == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = policy_noise * max_action
        kwargs["noise_clip"] = noise_clip * max_action
        kwargs["policy_freq"] = policy_freq
        policy = TD3.TD3(**kwargs)
    else:
        raise NotImplementedError()

    if load_model != "":
        policy_file = file_name if load_model == "default" else load_model
        policy.load(f"./models/{policy_file}")

    # choose replay buffer
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Initialize Agent with policy
    agent = Agent(action_dim, policy, replay_buffer, burnin=start_timesteps)

    # Evaluate untrained policy
    agent.eval_policy(env_name, seed)
    # delete: evaluations = [eval_policy(policy, env_name, seed)]

    # Main training loop
    # -----

    # reset environment
    state, done = env.reset(), False
    # state = np.concatenate([state[k] for k in state.keys()])

    # start logging of parameters
    # initialize logger
    # logger = utils.logger.TBMetricLogger()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    success = 0

    # training loop
    for t in range(int(max_timesteps)):

        episode_timesteps += 1

        # 3. Render environment [WIP]
        # env.render()

        # 4. Run agent on the state (get the action)
        # Select action randomly or according to policy

        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, max_action, expl_noise)

        # 5. Agent performs action
        next_state, reward, done, _ = env.step(action)
        # next_state = np.concatenate([next_state[k] for k in next_state.keys()])

        done_bool = float(
            done) if episode_timesteps < env._max_episode_steps else 0

        # 6. Remember
        # Store data in replay buffer
        agent.add_to_memory(state, action, next_state, reward, done_bool)

        # 9. Update state
        state = next_state
        episode_reward += reward

        # 8. Logging
        logger.log_step(reward, None, None)
        # 7. Learn
        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            agent.learn(batch_size, logger.writer)

        if done:
            if episode_timesteps < env._max_episode_steps:
                success += 1
                logger.writer.add_scalar(
                    'agent/cumulative_success', success, t)
            # +1 to account for 0 indexing. +0 on ep_timesteps
            # since it will increment +1 even if done=True
            print(" " * 80 + "\r" +
                  f"Total T: {t+1} Episode Num: {episode_num+1} \
                Episode T: {episode_timesteps} Reward: {episode_reward:.3f}",
                  end="\r")

            # Reset environment
            state, done = env.reset(), False
            # state = np.concatenate([state[k] for k in state.keys()])
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            # 10. Episode based metrics
            # Evaluate episode
            logger.log_episode(t)

        if (t + 1) % eval_freq == 0:
            agent.eval_policy(env_name, seed)
            # log evaluation results with non-noisy action selection
            logger.writer.add_scalar(
                'agent/eval_episode_reward', agent.evaluations[-1], t)
            agent.create_policy_eval_video(env_name, seed, results_dir + f"/t_{t+1}")
            if save_model:
                policy.save(f".{results_dir}/models/{file_name}")


def parse_args():
    parser = argparse.ArgumentParser()

    # Place your experiment arguments here
    arg_conf = parser.add_argument_group('Config')
    # Policy name (TD3, or else if implemented)
    arg_conf.add_argument("--policy_name", type=str)
    # OpenAI gym environment name
    arg_conf.add_argument("--env_name", type=str)
    # Time steps initial random policy is used
    arg_conf.add_argument("--start_timesteps", type=int)
    # How often (time steps) we evaluate
    arg_conf.add_argument("--eval_freq", type=int)
    # Max time steps to run environment
    arg_conf.add_argument("--max_timesteps", type=int)
    # Std of Gaussian exploration noise
    arg_conf.add_argument("--expl_noise")
    # Batch size for both actor and critic
    arg_conf.add_argument("--batch_size", type=int)
    arg_conf.add_argument("--discount")                 # Discount factor
    # Target network update rate
    arg_conf.add_argument("--tau")
    # Noise added to target policy during critic update
    arg_conf.add_argument("--policy_noise")
    # Range to clip target policy noise
    arg_conf.add_argument("--noise_clip")
    # Frequency of delayed policy updates
    arg_conf.add_argument("--policy_freq", type=int)
    # Save model and optimizer parameters
    arg_conf.add_argument("--save_model", action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    arg_conf.add_argument("--load_model", default="")

    # Leave unchanged
    parser = add_launcher_base_args(parser)
    parser.set_defaults(**get_default_params(experiment))
    args = parser.parse_args()
    return vars(args)


# ----------------
# main program
# ----------------

if __name__ == '__main__':
    # Leave unchanged
    args = parse_args()
    run_experiment(experiment, args)


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
