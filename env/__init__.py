#!/usr/bin/python
# _____________________________________________________________________________
import env.mujoco as emj
import panda_gym
import gym_robotics

#---

import sys

sys.path.append('../')
from gym.envs.registration import register
import gym
#from env.bitflip import BitFlipEnv
#from env.fourroom import FourRoom, FourRoom2, FourRoom3, FourRoom4
#from env.mountaincar import MountainCarEnv
#from env.plane import NaivePlane, NaivePlane2, NaivePlane3, NaivePlane4, NaivePlane5
#from env.nchain import NChainEnv
from env.goal_plane_env import GoalPlane

# register(
#     id='Bitflip-v0',
#     entry_point='env.bitflip:BitFlipEnv',
#     kwargs={'num_bits': 11},
#     max_episode_steps=200,
#     reward_threshold=100.0,
#     nondeterministic=False,
# )
# 
# N = 64
# register(
#     id='NChain-v1',
#     entry_point='env.nchain:NChainEnv',
#     kwargs={'n': N,
#             'slip': 0.1,
#             },
#     max_episode_steps=N+10,
# )
# 
# register(
#     id='FourRoom-v0',
#     entry_point='env.fourroom:FourRoom',
#     kwargs={'goal_type': 'fix_goal'},
#     max_episode_steps=200,
#     reward_threshold=100.0,
#     nondeterministic=False,
# )
# 
# register(
#     id='FourRoom-v1',
#     entry_point='env.fourroom:FourRoom2',
#     kwargs={'goal_type': 'fix_goal'},
#     max_episode_steps=200,
#     reward_threshold=100.0,
#     nondeterministic=False,
# )
# 
# register(
#     id='FourRoom-v2',
#     entry_point='env.fourroom:FourRoom3',
#     kwargs={'goal_type': 'fix_goal'},
#     max_episode_steps=200,
#     reward_threshold=100.0,
#     nondeterministic=False,
# )
# 
# register(
#     id='FourRoom-v4',
#     entry_point='env.fourroom:FourRoom4',
#     kwargs={'goal_type': 'fix_goal'},
#     max_episode_steps=200,
#     reward_threshold=100.0,
#     nondeterministic=False,
# )
# 
# register(
#     id='mcar-v0',
#     entry_point='env.mountaincar:MountainCarEnv',
#     kwargs={'goal_dim': 1},
#     max_episode_steps=200,
#     reward_threshold=100.0,
#     nondeterministic=False,
# )
# 
register(
    id='Plane-v0',
    entry_point='env.plane:NaivePlane5',
)


register(
    id='GoalPlane-v0',
    entry_point='env.goal_plane_env:GoalPlane',
    max_episode_steps=150,
    reward_threshold=195.0,
    kwargs={
        "env_name": "Plane-v0",
        "maze_size": 15,
        "action_size": 1,
        "distance": 1.,
        "start": (2.5, 2.5),
    }
)

register(
    id='GoalPlaneMid-v0',
    entry_point='env.goal_plane_env:GoalPlane',
    max_episode_steps=50,
    reward_threshold=195.0,
    kwargs={
        "env_name": "Plane-v0",
        "type": "mid",
        "maze_size": 15,
        "action_size": 1,
        "distance": 1.,
        "start": (2.5, 2.5),
    }
)

register(
    id='GoalPlaneHard-v0',
    entry_point='env.goal_plane_env:GoalPlane',
    max_episode_steps=50,
    reward_threshold=195.0,
    kwargs={
        "env_name": "Plane-v0",
        "type": "hard",
        "maze_size": 15,
        "action_size": 1,
        "distance": 1.,
        "start": (2.5, 2.5),
    }
)

register(
    id='GoalPlaneEasy-v0',
    entry_point='env.goal_plane_env:GoalPlane',
    max_episode_steps=50,
    reward_threshold=195.0,
    kwargs={
        "env_name": "Plane-v0",
        "type": "easy",
        "maze_size": 15,
        "action_size": 1,
        "distance": 1.,
        "start": (2.5, 2.5),
    }
)

register(
    id='GoalPlaneTest-v0',
    entry_point='env.goal_plane_env:GoalPlane',
    max_episode_steps=50,
    reward_threshold=195.0,
    kwargs={
        "env_name": "Plane-v0",
        "maze_size": 15,
        "action_size": 1,
        "distance": 1.,
        "start": (2.5, 2.5),
        "goals": (2.5, 12.5)
    }
)
