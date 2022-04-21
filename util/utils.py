#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
import numpy as np

# utilities
# -----
# from util import *

# custom functions
# -----

class SubgoalActionSpace(object):
    def __init__(self, dim):
        limits = np.array([-10, -10, -0.5, -1, -1, -1, -1,
                    -0.5, -0.3, -0.5, -0.3, -0.5, -0.3, -0.5, -0.3])
        self.shape = (dim,1)
        self.low = limits[:dim]
        self.high = -self.low

    def sample(self):
        return (self.high - self.low) * np.random.sample(self.high.shape) + self.low

class Subgoal(object):
    def __init__(self, dim=15):
        self.action_space = SubgoalActionSpace(dim)
        self.action_dim = self.action_space.shape[0]


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
