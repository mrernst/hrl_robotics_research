#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import numpy as np
import torch



# custom classes
# -----

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, sequence_dim, offpolicy, max_size=int(1e6)):
                
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        sequence_dim = sequence_dim if offpolicy else 1
        self.state_seq = np.zeros((max_size, sequence_dim, state_dim))  
        self.action_seq = np.zeros((max_size, sequence_dim, action_dim))
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state_seq, action_seq, state, action, next_state, reward, done):
        self.state_seq[self.ptr] = state_seq
        self.action_seq[self.ptr] = action_seq
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = _get_indices(size=batch_size)

        return (
            torch.FloatTensor(self.state_seq[ind]).to(self.device),
            torch.FloatTensor(self.action_seq[ind]).to(self.device),
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
    def _get_indices(self, size):
        return np.random.randint(0, self.size, size=size)


class PERBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay
    """
    def __init__(self, ):
        raise NotImplementedError("Prioritized Experience Replay is not yet implemented")
    
    def add():
        pass
    
    def sample():
        pass

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
