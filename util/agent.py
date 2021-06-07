import torch
import random, numpy as np
from pathlib import Path

from util.neuralnet import *
from collections import deque

class QLearningAgent():
    def __init__(self, state_dim, action_dim, save_dir, memory_size=10000, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=memory_size)
        
        # configuration
        self.batch_size = 32
        self.exploration_rate = 0.9
        self.exploration_rate_decay = 0.99999
        self.exploration_rate_min = 0.1
        self.gamma = 0.99
        
        self.curr_step = 0
        self.burnin = 1024 #1e5 # min. experiences before training
        self.learn_every = 1 # no. of experiences between updates to Q_online
        self.sync_every = 1e3 #1e4   # no. of experiences between Q_target & Q_online sync
        
        self.save_every = 5e4   # no. of experiences between saving Mario Net
        self.save_dir = save_dir
        
        self.use_cuda = torch.cuda.is_available()
        
        self.neuralnet = QNetwork(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.neuralnet = self.neuralnet.to(device='cuda')
        if checkpoint:
            self.load(checkpoint)
        
        self.optimizer = torch.optim.Adam(self.neuralnet.parameters(), lr=0.001)
        #self.loss_fn = torch.nn.SmoothL1Loss()
        self.loss_fn = torch.nn.MSELoss()

    
    def act(self, state):
        # exploration in the environment
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # exploitation of the environment
        else:
            state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
            state = state.unsqueeze(0)
            action_values = self.neuralnet(state, model='online')
            action_idx = torch.argmax(action_values, axis=1).item()
        
        # decay of exploration rate
        if self.curr_step > self.burnin:
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        
        # increment step
        self.curr_step += 1
	    
        return action_idx
    
    
    def add_to_memory(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        
        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = torch.FloatTensor(state).cuda() if self.use_cuda else torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state).cuda() if self.use_cuda else torch.FloatTensor(next_state)
        action = torch.LongTensor([action]).cuda() if self.use_cuda else torch.LongTensor([action])
        reward = torch.DoubleTensor([reward]).cuda() if self.use_cuda else torch.DoubleTensor([reward])
        done = torch.BoolTensor([done]).cuda() if self.use_cuda else torch.BoolTensor([done])
        
        self.memory.append( (state, next_state, action, reward, done,) )
        	
        pass
    
    def sample_from_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
        
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_tarQ()
        
        if self.curr_step % self.save_every == 0:
            self.save()
        
        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.learn_every != 0:
            return None, None
        
        # Sample from memory
        state, next_state, action, reward, done = self.sample_from_memory()
        
        # TD Estimate
        td_est = self.neuralnet(state, model='online')[np.arange(0, self.batch_size), action]
        
        # TD Target
        with torch.no_grad():
            next_state_Q = self.neuralnet(next_state, model='online')
            best_action = torch.argmax(next_state_Q, axis=1)
            next_Q = self.neuralnet(next_state, model='target')[np.arange(0, self.batch_size), best_action]
            td_tgt = (reward + (1 - done.float()) * self.gamma * next_Q).float()
        
        
            
        # Backpropagate through the online network Q_online
        loss = self.loss_fn(td_est, td_tgt)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return (td_est.mean().item(), loss.item())
        
    def save(self):
        save_path = self.save_dir / f"DQN_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.neuralnet.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"DQN saved to {save_path} at step {self.curr_step}")
        pass
        
    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")
        
        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')
        
        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.neuralnet.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
        pass
    
    def sync_tarQ(self):
    	self.neuralnet.target.load_state_dict(
    		self.neuralnet.online.state_dict()
    	)
    	pass



class PPOAgent():
    """To be implemented"""
    pass
