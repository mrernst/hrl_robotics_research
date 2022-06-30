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
        self._append_callback()
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        self.ind = self._get_indices(size=self.batch_size)

        return (
            torch.FloatTensor(self.state[self.ind]).to(self.device),
            torch.FloatTensor(self.goal[self.ind]).to(self.device),
            torch.FloatTensor(self.action[self.ind]).to(self.device),
            torch.FloatTensor(self.n_state[self.ind]).to(self.device),
            torch.FloatTensor(self.reward[self.ind]).to(self.device),
            torch.FloatTensor(self.not_done[self.ind]).to(self.device),
        )
    
    def _get_indices(self, size):
        return np.random.randint(0, self.size, size=size)
    
    def _append_callback(self):
        pass



class PERReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, goal_dim, action_dim, buffer_size, batch_size):
        super(PERReplayBuffer, self).__init__(state_dim, goal_dim, action_dim, buffer_size, batch_size)
        self.epsilon = 0.01
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self._per_error_type = 1
        
        self.tree = SumTree(buffer_size)
        self.priorities = np.zeros((buffer_size, 1), dtype = np.float32)

    @property
    def per_error_type(self):
        return self._per_error_type
    
    @per_error_type.setter
    def per_error_type(self, n):
        self._per_error_type = n

    def _get_priority(self, error):
        '''
        Takes in the error of one or more examples and returns the proportional priority
        '''
        return np.power(np.abs(error) + self.epsilon, self.alpha).squeeze()
    
    
    def _get_indices(self, size):
        '''
        Samples batch_size indices from memory in proportional to their priority.
        '''
        batch_idxs = np.zeros(size)
        tree_idxs = np.zeros(size, dtype=np.int64)
        priorities = np.zeros([size, 1]) 
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        segment = self.tree.total() / size
        
        for i in range(size):
            # segments are a problem if batchsize if total size is non-dividable
            # a = segment * i
            # b = segment * (i + 1)
            # s = np.random.uniform(a, b)
            s = np.random.uniform(0, self.tree.total())
            (tree_idx, p, idx) = self.tree.get(s)
            batch_idxs[i] = idx
            tree_idxs[i] = tree_idx
            priorities[i] = p
        
        batch_idxs = np.asarray(batch_idxs).astype(int)
        self.tree_idxs = tree_idxs
        
        sampling_probabilities = priorities / self.tree.total()
        self.is_weight = np.power(self.tree.n_entries * sampling_probabilities, - self.beta)
        self.is_weight /= self.is_weight.max()
        
        return batch_idxs
    
    def _append_callback(self, error=100000):
        '''
        Callback function that is called at the beginning of sample
        It is used here to store the priorities of the samples into the SumTree object
        '''
        priority = self._get_priority(error)
        self.priorities[self.ptr] = priority
        self.tree.add(priority, self.ptr)
    
    def update(self, errors):
        '''
        Updates the priorities from the most recent batch
        Assumes the relevant batch indices are stored in self.ind
        '''
        priorities = self._get_priority(errors)
        assert len(priorities) == self.ind.size
        for idx, p in zip(self.ind, priorities):
            self.priorities[idx] = p
        for p, i in zip(priorities, self.tree_idxs):
            self.tree.update(i, p)
    

    
class SumTree:
    '''
    Taken from 'Foundations of deep reinforcement learning' Book:
    Helper class for PrioritizedReplay
    This implementation is, with minor adaptations, Jaromír Janisch's. The license is reproduced below.
    For more information see his excellent blog series "Let's make a DQN" https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
    MIT License
    Copyright (c) 2018 Jaromír Janisch
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    
    In this implementation, a conventional FIFO buffer holds the data, while the tree 
    only holds the indices and the corresponding priorities.
    '''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Stores the priorities and sums of priorities
        self.indices = np.zeros(capacity)  # Stores the indices of the experiences
        self.n_entries = 0.

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, index):
        idx = self.write + self.capacity - 1

        self.indices[self.write] = index
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        assert s <= self.total()
        idx = self._retrieve(0, s)
        indexIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.indices[indexIdx])

    def print_tree(self):
        for i in range(len(self.indices)):
            j = i + self.capacity - 1
            print(f'Idx: {i}, Data idx: {self.indices[i]}, Prio: {self.tree[j]}')



class LowLevelReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, goal_dim, action_dim, buffer_size, batch_size):
        super(LowLevelReplayBuffer, self).__init__(state_dim, goal_dim, action_dim, buffer_size, batch_size)
        self.n_goal = np.zeros((buffer_size, goal_dim))

    def append(self, state, goal, action, n_state, n_goal, reward, done):
        self._append_callback()
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
        self.ind = self._get_indices(size=self.batch_size)

        return (
            torch.FloatTensor(self.state[self.ind]).to(self.device),
            torch.FloatTensor(self.goal[self.ind]).to(self.device),
            torch.FloatTensor(self.action[self.ind]).to(self.device),
            torch.FloatTensor(self.n_state[self.ind]).to(self.device),
            torch.FloatTensor(self.n_goal[self.ind]).to(self.device),
            torch.FloatTensor(self.reward[self.ind]).to(self.device),
            torch.FloatTensor(self.not_done[self.ind]).to(self.device),
        )

class HighLevelReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, goal_dim, subgoal_dim, action_dim, buffer_size, batch_size, freq):
        super(HighLevelReplayBuffer, self).__init__(state_dim, goal_dim, action_dim, buffer_size, batch_size)
        self.action = np.zeros((buffer_size, subgoal_dim))
        self.state_arr = np.zeros((buffer_size, freq, state_dim))
        self.action_arr = np.zeros((buffer_size, freq, action_dim))

    def append(self, state, goal, action, n_state, reward, done, state_arr, action_arr):
        self._append_callback()
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
        self.ind = self._get_indices(size=self.batch_size)
        
        return (
            torch.FloatTensor(self.state[self.ind]).to(self.device),
            torch.FloatTensor(self.goal[self.ind]).to(self.device),
            torch.FloatTensor(self.action[self.ind]).to(self.device),
            torch.FloatTensor(self.n_state[self.ind]).to(self.device),
            torch.FloatTensor(self.reward[self.ind]).to(self.device),
            torch.FloatTensor(self.not_done[self.ind]).to(self.device),
            torch.FloatTensor(self.state_arr[self.ind]).to(self.device),
            torch.FloatTensor(self.action_arr[self.ind]).to(self.device)
        )


class LowLevelPERReplayBuffer(LowLevelReplayBuffer, PERReplayBuffer):
    pass

class HighLevelPERReplayBuffer(HighLevelReplayBuffer, PERReplayBuffer):
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
