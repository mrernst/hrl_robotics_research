#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
import numpy as np

import gym
from collections import OrderedDict

# utilities
# -----
from env.mujoco.maze_env_utils import construct_maze

# constants

HIRO_DEFAULT_LIMITS = np.array(
        # COM position (m)
        [-10, -10, -0.5,
        # Orientation quaternion (au)
        -1, -1, -1, -1,
        # Joint positions (rad)
        -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3,
        # TODO: find plausible ranges here
        #COM velocity (m/s)
        -1,-1,-1,
        #Change in orientation (dw/dt)
        -1,-1,-1,
        #Joint velocities (rad/s)
        -1,-1,-1,-1,-1,-1,-1,-1,
        #Target position (m)
        -10,-10,
], dtype=np.float32)

ANTENV_STATE_NAMES = np.array([
    'com_pos_x', 'com_pos_y', 'com_pos_z',
    'oq1', 'oq2', 'oq3', 'oq4',
    'joint_pos_1', 'joint_pos_2','joint_pos_3','joint_pos4','joint_pos_5','joint_pos_6','joint_pos_7','joint_pos_8',
    'com_vel_x', 'com_vel_y', 'com_vel_z',
    'dw1/dt', 'dw2/dt', 'dw3/dt',
    'joint_vel_1', 'joint_vel_2', 'joint_vel_3', 'joint_vel_4', 'joint_vel_5', 'joint_vel_6', 'joint_vel_7', 'joint_vel_8',
    'target_pos_x', 'target_pos_y',]
)

# custom functions
# -----

def mkdir_p(path):
    """
    Takes a string path and creates a directory at this path if it
    does not already exist.
    params:
        path: path to be created
    return:
        None
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    

def print_tensor_info(tensor, name=None, writer=None):
    """
    Returns name and shape of a torch.Tensor for debugging purposes.
    params:
        tensor: the torch.Tensor you are interested in (torch.Tensor)
        name: name for logging purposes (str)
        writer: torch.tensorboard_writer object for logging
    return:
        None
    """
    
    if writer:
        writer.add_scalar(name + '/min', tensor.min(), global_step=global_step)
        writer.add_scalar(name + '/max', tensor.max(), global_step=global_step)
        writer.add_scalar(name + '/std', tensor.type(torch.float).std(), global_step=global_step)
        writer.add_scalar(name + '/mean', tensor.type(torch.float).mean(), global_step=global_step)
    else:
        name = name if name else tensor.names
        text = "[DEBUG] name = {}, shape = {}, dtype = {}, device = {} \n" + \
        "\t min = {}, max = {}, std = {}, mean = {}"
        print(text.format(name, list(tensor.shape), tensor.dtype, tensor.device.type,
        tensor.min(), tensor.max(), tensor.type(torch.float).std(), tensor.type(torch.float).mean()))
        
    pass


def random_sample(size=None, dtype=np.float64):
    """
    np.random.random_sample but with variable precision to better work with torch
    TODO: make everything work with Tensors and only use numpy for analysis
    """
    type_max = 1 << np.finfo(dtype).nmant
    sample = np.empty(size, dtype=dtype)
    sample[...] = np.random.randint(0, type_max, size=size) / dtype(type_max)
    if size is None:
        sample = sample[()]
    return sample


def _is_update(episode, freq, ignore=0, rem=0):
        if episode!=ignore and episode%freq==rem:
            return True
        return False
    
    
def get_obs_array(state, combined=False):
    try:
        if combined:
            np.concatenate([state[k] for k in state.keys()])
        else:
            state = state['observation']
    except:
        pass
    
    return state


# custom classes
# -----

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GoalActionSpace(object):
    def __init__(self, dim, limits):
        self.shape = (dim,1)
        self.low = limits[:dim]
        self.high = -self.low

    def sample(self): 
        sample = (self.high - self.low) * random_sample(self.high.shape, dtype=np.float32) + self.low
        return sample
    
    def sample_batch(self, batch_size):
        sample = (self.high - self.low) * random_sample([batch_size] + list(self.high.shape), dtype=np.float32) + self.low
        return sample

class Subgoal(object):
    def __init__(self, dim=15, limits=HIRO_DEFAULT_LIMITS):
        self.action_space = GoalActionSpace(dim, limits)
        self.action_dim = self.action_space.shape[0]
    def __str__(self):
        return f"""Subgoal object, shape={self.action_space.shape}
                    \n\t action_space low:{self.action_space.low}
                    \n\t action_space high:{self.action_space.high}
                    """





def visualize_goalspace(grid, state_encoder, state_trajectory, goal_trajectory):
    raise NotImplementedError("To be done.")
    # this function is planned take a grid within the 2D or 3D position space of the environment
    # (may be different for a different environment) and uses the state encoder network to encode
    # and then visualizes the grid using a pacmap with euclidean distance encoded with color
    # could also draw the path taken in the subgoal space
    fig = None
    return fig


# def visualize_eval_trajectories(env_name, overlay=True):
#     maze_structure = construct_maze(env_name)
#     size = 2,4,8 for v2, v1, v0
#     pass

def visualize_endpoints():
    raise NotImplementedError("To be done.")
    pass

def draw_2d_env_map():
    raise NotImplementedError("To be done.")
    pass
    
    
def _compose_alpha(img_in, img_layer, opacity):
    """
    Calculate alpha composition ratio between two images.
    """
    
    comp_alpha = np.minimum(img_in[:, :, 3], img_layer[:, :, 3]) * opacity
    new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3]) * comp_alpha
    np.seterr(divide='ignore', invalid='ignore')
    ratio = comp_alpha / new_alpha
    ratio[ratio == np.nan] = 0.0
    return ratio

def darken(img_in, img_layer, opacity):
    """
    Apply darken only blending mode of a layer on an image.
    """
        
    img_in_norm = img_in / 255.0
    img_layer_norm = img_layer / 255.0
    
    ratio = _compose_alpha(img_in_norm, img_layer_norm, opacity)
    
    comp = np.minimum(img_in_norm[:, :, :3], img_layer_norm[:, :, :3])
    
    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in_norm[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in_norm[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0

# wrapper to make standard environments 'goal-based' by providing a 
# fake goal in order to test the flat agent
class MakeGoalBased(gym.Wrapper):
    def __init__(self, env):
        super(MakeGoalBased, self).__init__(env)
        ob_space = env.observation_space
        self.goal_space = gym.spaces.Box(low=np.array([0,0]), high=np.array([1,1]))
        self.observation_space = gym.spaces.Dict(OrderedDict({
            'observation': ob_space,
            'desired_goal': self.goal_space,
            'achieved_goal': self.goal_space,
        }))
        self._max_episode_steps = self.env._max_episode_steps
    def step(self, action):
        #print(action, action.shape)
        observation, reward, done, info = self.env.step(action)
        out = {'observation': observation,
               'desired_goal': np.array([0,0]),
               'achieved_goal': np.array([0,0])}
        return out, reward, done, info
    
    def reset(self):
        observation = self.env.reset()
    
        out = {'observation': observation,
               'desired_goal': np.array([0,0]),
               'achieved_goal': np.array([0,0])}
        return out


# custom classes
# -----


# ----------------
# main program
# ----------------

if __name__ == "__main__":
    pass


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
