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

# class ReplayBuffer(object):
#     def __init__(self, state_dim, action_dim, sequence_dim, offpolicy, max_size=int(1e6)):
#                 
#         self.max_size = max_size
#         self.ptr = 0
#         self.size = 0
#         
#         sequence_dim = sequence_dim if offpolicy else 1
#         self.state_seq = np.zeros((max_size, sequence_dim, state_dim))  
#         self.action_seq = np.zeros((max_size, sequence_dim, action_dim))
#         
#         self.state = np.zeros((max_size, state_dim))
#         self.action = np.zeros((max_size, action_dim))
#         self.next_state = np.zeros((max_size, state_dim))
#         self.reward = np.zeros((max_size, 1))
#         self.not_done = np.zeros((max_size, 1))
# 
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu")
# 
#     def add(self, state_seq, action_seq, state, action, next_state, reward, done):
#         self.state_seq[self.ptr] = state_seq
#         self.action_seq[self.ptr] = action_seq
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.next_state[self.ptr] = next_state
#         self.reward[self.ptr] = reward
#         self.not_done[self.ptr] = 1. - done
# 
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
# 
#     def sample(self, batch_size):
#         ind = self._get_indices(size=batch_size)
# 
#         return (
#             torch.FloatTensor(self.state_seq[ind]).to(self.device),
#             torch.FloatTensor(self.action_seq[ind]).to(self.device),
#             torch.FloatTensor(self.state[ind]).to(self.device),
#             torch.FloatTensor(self.action[ind]).to(self.device),
#             torch.FloatTensor(self.next_state[ind]).to(self.device),
#             torch.FloatTensor(self.reward[ind]).to(self.device),
#             torch.FloatTensor(self.not_done[ind]).to(self.device)
#         )
#     
#     def _get_indices(self, size):
#         return np.random.randint(0, self.size, size=size)




class ReplayBuffer(object):
    def __init__(self, state_dim, goal_dim, action_dim, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((buffer_size, state_dim))
        self.goal = np.zeros((buffer_size, goal_dim))
        self.action = np.zeros((buffer_size, action_dim))
        self.n_state = np.zeros((buffer_size, state_dim))
        self.reward = np.zeros((buffer_size, 1))
        self.not_done = np.zeros((buffer_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def append(self, state, goal, action, n_state, reward, done):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        ind = self._get_indices(size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )
    
    def _get_indices(self, size):
        return np.random.randint(0, self.size, size=size)

class LowLevelReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, goal_dim, action_dim, buffer_size, batch_size):
        super(LowLevelReplayBuffer, self).__init__(state_dim, goal_dim, action_dim, buffer_size, batch_size)
        self.n_goal = np.zeros((buffer_size, goal_dim))

    def append(self, state, goal, action, n_state, n_goal, reward, done):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.n_goal[self.ptr] = n_goal
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        ind = self._get_indices(size=self.batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.n_goal[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

class HighLevelReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, goal_dim, subgoal_dim, action_dim, buffer_size, batch_size, freq):
        super(HighLevelReplayBuffer, self).__init__(state_dim, goal_dim, action_dim, buffer_size, batch_size)
        self.action = np.zeros((buffer_size, subgoal_dim))
        self.state_arr = np.zeros((buffer_size, freq, state_dim))
        self.action_arr = np.zeros((buffer_size, freq, action_dim))

    def append(self, state, goal, action, n_state, reward, done, state_arr, action_arr):
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.state_arr[self.ptr,:,:] = state_arr
        self.action_arr[self.ptr,:,:] = action_arr

        self.ptr = (self.ptr+1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)

    def sample(self):
        ind = self._get_indices(size=self.batch_size)
        
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.goal[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.n_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.state_arr[ind]).to(self.device),
            torch.FloatTensor(self.action_arr[ind]).to(self.device)
        )


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
