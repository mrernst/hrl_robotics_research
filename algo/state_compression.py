#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import lstsq


from algo.td3 import get_tensor
from util.utils import HIRO_DEFAULT_LIMITS, ANTENV_STATE_NAMES, GoalActionSpace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# utilities
# -----

# similarity functions
def cosine_sim(x, x_pair):
    """
        Similarity function based on cosine similarity
        params:
            x: (torch.Tensor)
            x_pair: (torch.Tensor)
        return:
            sim: the similarity between x and x_pair (torch.Tensor)
    """
    return F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0), dim=2)

def rbf_sim(x, x_pair):
    """
        Similarity function based on radial basis functions
        params:
            x: (torch.Tensor)
            x_pair: (torch.Tensor)
        return:
            sim: the similarity between x and x_pair (torch.Tensor)
    """
    return -torch.cdist(x, x_pair)


@torch.no_grad()
def lls_fit(train_features, train_labels):
    """
        Fit a linear least square model
        params:
            train_features: the representations to be trained on (Tensor)
            train_labels: labels of the original data (LongTensor)
            n_classes: int, number of classes
        return:
            ls: the trained lstsq model (torch.linalg) 
    """
    ls = lstsq(train_features, train_labels)
    
    return ls
    
@torch.no_grad()
def lls_eval(trained_lstsq_model, eval_features):
    """
        Evaluate a trained linear least square model
        params:
            trained_lstsq_model: the trained lstsq model (torch.linalg)
            eval_features: the representations to be evaluated on (Tensor)
            eval_labels: labels of the data (LongTensor)
        return:
            acc: the LLS accuracy (float)
    """
    prediction = (eval_features @ trained_lstsq_model.solution)
    return prediction

# custom classes
# -----



class SimCLR_TT_Loss(nn.Module):
    def __init__(self, sim_func, batch_size, temperature):
        """Initialize the SimCLR_TT_Loss class"""
        super(SimCLR_TT_Loss, self).__init__()
    
        self.batch_size = batch_size
        self.temperature = temperature
    
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.sim_func = sim_func
    
    def mask_correlated_samples(self, batch_size):
        """
        mask_correlated_samples takes the int batch_size
        and returns an np.array of size [2*batchsize, 2*batchsize]
        which masks the entries that are the same image or
        the corresponding positive contrast
        """
        mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
    
        # fill off-diagonals corresponding to positive samples
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    
    def forward(self, x, x_pair, labels=None, N_negative=None):
        """
        Given a positive pair, we treat the other 2(N − 1)
        augmented examples within a minibatch as negative examples.
        to control for negative samples we just cut off losses
        """
        N = 2 * self.batch_size
    
        z = torch.cat((x, x_pair), dim=0)
    
        sim = self.sim_func(z, z) / self.temperature
    
        # get the entries corresponding to the positive pairs
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    
        # we take all of the negative samples
        negative_samples = sim[self.mask].reshape(N, -1)
    
        if N_negative:
            # if we specify N negative samples: do random permutation of negative sample losses,
            # such that we do consider different positions in the batch
            negative_samples = torch.take_along_dim(
                negative_samples, torch.rand(*negative_samples.shape, device=device).argsort(dim=1), dim=1)
            # cut off array to only consider N_negative samples per positive pair
            negative_samples = negative_samples[:, :N_negative]
            # so what we are doing here is basically using the batch to sample N negative
            # samples.
    
        # the following is more or less a trick to reuse the cross-entropy function for the loss
        # Think of the loss as a multi-class problem and the label is 0
        # such that only the positive similarities are picked for the numerator
        # and everything else is picked for the denominator in cross-entropy
        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
    
        return loss




# compressor classes
# -----


class StateCompressor(object):
    """
    StateCompressor pseudoclass
    """
    def __init__(self, subgoal_dim):
        #self.subgoal = Subgoal(subgoal_dim)
        self.subgoal_dim = subgoal_dim
        self.curr_train_metrics = {}
    
    def __call__(self, state):
        raise NotImplementedError("StateCompressor is a pseudoclass, inherit from it and implement the forward function")
        pass


class SliceCompressor(StateCompressor):
    """
    SliceCompressor is a StateCompressor class that compresses the state by just slicing the original state to the dimensionality defined as subgoal_dim
    """
    def __init__(self, subgoal_dim):
        super().__init__(subgoal_dim)
    
    def __call__(self, state):
        return state[:self.subgoal_dim]
    
    def eval(self, state):
        return self(state)


class NetworkCompressor(nn.Module):
    """
    NetworkCompressor is a StateCompressor class that compresses the state by funneling it through a neural network. Depending on the network class this can be an Encoder or Autoencoder structure
    """
    
    def __init__(self, network, loss_fn, learning_rate, weight_decay, name='state_compressor'):
        
        super(NetworkCompressor, self).__init__()
        self.network = network
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.loss_fn = loss_fn
        
        self.state_space = GoalActionSpace(dim=29, limits=HIRO_DEFAULT_LIMITS)
        
        self.curr_train_metrics = {}
        self.name = name
        
    def eval(self, state):
        # activate evaluation mode
        self.network.eval()
        
        state = get_tensor(state)
        with torch.no_grad():
            representation, projection = self.network(state)
        
        # return to training state
        self.network.train()
        return representation.cpu()
    

    def forward(self, state):
        representation, projection = self.network(state)
        return representation, projection
    
    def _collect_metrics(self, loss):
        with torch.no_grad():
            norm_type = 2
            gr_norm = torch.norm(torch.stack([torch.norm(
                p.grad.detach(), norm_type) for p in self.network.parameters()]), norm_type)
            self.curr_train_metrics['gradients/norm'] = gr_norm
            self.curr_train_metrics['loss'] = loss
    
    def _plot_state_correlation(self, inp, projection):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import scipy as sp
        
        def annotate_correlation(data, **kws):
            r, p = sp.stats.pearsonr(data['input'], data['projection'])
            ax = plt.gca()
            ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                    transform=ax.transAxes)
                    
        df_list = []
        for i in range(inp.shape[1]):
            df_list.append(pd.DataFrame({'input': inp[:,i], 'projection': projection[:,i], "type":np.repeat(ANTENV_STATE_NAMES[i], inp.shape[0])}))
        
        df = pd.concat(df_list, ignore_index=True)
        
        sns.set_theme(style="ticks", font_scale=0.65)
                # Show the results of a linear regression within each dataset
        p = sns.lmplot(x="input", y="projection", col="type", hue="type", data=df,
                   col_wrap=5, sharex=False, sharey=False, ci=None, palette="muted", height=1.25,
                   scatter_kws={"s": 2, "alpha": 1})
        p.map_dataframe(annotate_correlation)
        #plt.show()
        return p.fig
        


class EncoderCompressor(NetworkCompressor):
    def train(self, buffer, time_horizon):
        # Sample from the buffer
        x, x_pair = buffer.sample_with_timecontrast(time_horizon)
        
        x = torch.cat([x[0], x_pair[0]], 0).to(device)
        representation, projection = self.network(x)
        projection, pair = projection.split(projection.shape[0]//2)
        # Get the loss
        loss = self.loss_fn(projection, pair)
        # Optimize
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        # Collect metrics
        self._collect_metrics(loss)
        
        return loss
    
    def eval_state_info(self, batch_size, buffer=None):
        if buffer:
            inp = buffer.sample()[0]
        else:
            inp = self.state_space.sample_batch(batch_size)
            inp = get_tensor(inp)
        with torch.no_grad():
            representation, projection = self.forward(inp)
        
        lls_model = lls_fit(representation,inp)

        # use the "reconstruction" based on the lls
        reconstruction = lls_eval(lls_model, representation)

        fig = self._plot_state_correlation(inp.cpu(), reconstruction.cpu())
        
        return fig
        
class AutoEncoderCompressor(NetworkCompressor):
    def train(self, buffer, time_horizon):
        
        # Sample from the buffer
        x = buffer.sample()[0]
        representation, projection = self.network(x)
        
        # Get the loss
        loss = self.loss_fn(projection, x)
        pass
        
        # Optimize
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        
        # Collect metrics
        self._collect_metrics(loss)
        
        return loss
    
    def eval_state_info(self, batch_size, buffer=None):
        if buffer:
            inp = buffer.sample()[0]
        else:
            inp = self.state_space.sample_batch(batch_size)
            inp = get_tensor(inp)
        with torch.no_grad():
            representation, reconstruction = self.forward(inp)
        
        fig = self._plot_state_correlation(inp.cpu(), reconstruction.cpu())
        return fig
        



# encoder network structures
# -----

class EncoderNetwork(nn.Module):
    def __init__(self, state_dim, subgoal_dim, hidden_layers=[128]):
        super(EncoderNetwork, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], subgoal_dim)
                
    def forward(self, state):
        representation = F.relu(self.l1(state))
        representation = torch.tanh(self.l2(representation))
        # simple structure where representation and projection are the same
        return representation, representation


class AutoEncoderNetwork(nn.Module):
    def __init__(self, state_dim, subgoal_dim, hidden_layers=[128]):
        super(AutoEncoderNetwork, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], subgoal_dim)
        self.l3 = nn.Linear(subgoal_dim, hidden_layers[0])
        self.l4 = nn.Linear(hidden_layers[0], state_dim)
                
    def forward(self, state):
        representation = F.relu(self.l1(state))
        representation = torch.tanh(self.l2(representation))
        projection = F.relu(self.l3(representation))
        #projection = torch.tanh(self.l4(projection))
        projection = self.l4(projection)

        return representation, projection



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
