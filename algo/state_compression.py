#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import numpy as np
import os
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
    """
    Loss function for SimCLR-style contrastive learning
    """
    def __init__(self, sim_func, batch_size, temperature):
        """
        Initialize the SimCLR_TT_Loss class.
        params:
            sim_func: A similarity function like cosine-sim
            batch_size: The batch-size, needed for reducing negative samples
            temperature: The annealing temperature of the NCELoss
        return:
            None
        """
        super(SimCLR_TT_Loss, self).__init__()
    
        self.batch_size = batch_size
        self.temperature = temperature
    
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.sim_func = sim_func
    
    def mask_correlated_samples(self, batch_size):
        """
        Masks the entries that are the same image or the corresponding
        positive contrast.
        params:
            batch_size: The batch-size of the training
        return:
            mask: np.array of size [2*batch_size, 2*batch_size]
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
        To control for negative samples we just cut off losses.
        params:
            x: The 'first' contrast
            x_pair: The 'second' contrast
            labels: Can be introduced for a semi-supervised learning scheme
            N_negative: Limit of the number of negative pairs
        return:
            loss: The scalar NCE-Loss
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
    StateCompressor pseudoclass.
    """
    def __init__(self, subgoal_dim):
        """
        Initialize a StateCompressor class.
        params:
            subgoal_dim: dimension of the subgoal (int)
        return:
            None
        """
        #self.subgoal = Subgoal(subgoal_dim)
        self.subgoal_dim = subgoal_dim
        self.curr_train_metrics = {}
    
    def __call__(self, state):
        raise NotImplementedError("StateCompressor is a pseudoclass, inherit from it and implement the forward function")
        pass


class SliceCompressor(StateCompressor):
    """
    SliceCompressor is a StateCompressor class that compresses the state by slicing the original state to the dimensionality defined by subgoal_dim
    """
    def __init__(self, subgoal_dim):
        """
        Initialize a SliceCompressor class.
        params:
            subgoal_dim: dimension of the subgoal (int)
        return:
            None
        """
        super().__init__(subgoal_dim)
    
    def __call__(self, state):
        """
        Redefines a call on the object.
        params:
            state: The original state as an input
        return:
            sliced_state: A truncated state with dimensionality subgoal_dim
        """
        return state[:self.subgoal_dim]
    
    def eval(self, state):
        """
        Evaluate a given state. Redefinition of __call__ in order to
        guarantee compatibility with a torch.nn.Module object that
        uses the forward method to redefine __call__.
        params:
            state: The original state as an input
        return:
            sliced_state: A truncated state with dimensionality subgoal_dim
        """
        return self(state)


class NetworkCompressor(nn.Module):
    """
    NetworkCompressor is a StateCompressor class that compresses the state by funnelling it through a neural network. Depending on the child class this can be an Encoder or Auto-Encoder structure.
    """
    
    def __init__(self, model_path, network, loss_fn, learning_rate, weight_decay, name='state_compressor'):
        """
        Initialize a NetworkCompressor class.
        params:
            network: Neural network that returns two outputs (nn.Module)
            loss_fn: loss function that takes two inputs and returns a scalar value
            learning_rate: learning rate of the Adam optimizer used  (float)
            weight_decay: weight decay of the Adam optimizer used (float)
            name: The name of the module, used for logging (string)
        return:
            None
        """
        
        super(NetworkCompressor, self).__init__()
        self.network = network
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.loss_fn = loss_fn
        
        self.state_space = GoalActionSpace(dim=29, limits=HIRO_DEFAULT_LIMITS)
        
        self.curr_train_metrics = {}
        self.model_path = model_path
        self.name = name
        
    def eval(self, state):
        """
        Evaluate (compress) a given state.
        params:
            state: The original state as an input
        return:
            compressed_state: The output of the neural network
                compressing the state
        """
        # activate evaluation mode
        self.network.eval()
        
        state = get_tensor(state)
        with torch.no_grad():
            representation, projection = self.network(state)
        
        # return to training state
        self.network.train()
        return representation.cpu()
    

    def forward(self, state):
        """
        The forward pass through the network.
        params:
            state:  The original state as an input
        return:
            representation: The internal, compressed representation of
                the state
            projection: The representation that is used to optimize
        """
        representation, projection = self.network(state)
        return representation, projection
    
    def save(self, episode):
        """
        Save the network structure to disk.
        params:
            episode: current episode (int)
        return:
            None
        """
        # create episode directory. (e.g. model/2000)
        model_path = os.path.join(self.model_path, str(episode))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        torch.save(self.network.state_dict(),
            os.path.join(model_path, self.name+"_network"))
        torch.save(self.optimizer.state_dict(),
            os.path.join(model_path, self.name+"_optimizer"))

    
    def load(self, episode):
        """
        Load the network structure from disk.
        params:
            episode: episode number to load (int). Value of -1 loads 
                most recent save file
        return:
            None
        """
        if episode<0:
            episode_list = map(int, os.listdir(self.model_path))
            episode = max(episode_list)
        
        print(" " * 80 + "\r" + f'[INFO] Loaded model at episode {episode}')
        
        model_path = os.path.join(self.model_path, str(episode))
        
        self.network.load_state_dict(torch.load(os.path.join(
            model_path, self.name+"_network"), map_location=torch.device(device)))
        self.optimizer.load_state_dict(
            torch.load(os.path.join(model_path, self.name+"_optimizer"), map_location=torch.device(device)))
        
    
    def _collect_metrics(self, loss):
        """
        Helper-method to collect useful metrics for logging.
        params:
            loss: The current training loss (float)
        return:
            None
        """
        with torch.no_grad():
            norm_type = 2
            gr_norm = torch.norm(torch.stack([torch.norm(
                p.grad.detach(), norm_type) for p in self.network.parameters()]), norm_type)
            self.curr_train_metrics['gradients/norm'] = gr_norm
            self.curr_train_metrics['loss'] = loss
    
    
    def _plot_state_correlation(self, inp, projection):
        """
        Helper-method to plot the correlation between input and projected output states, which yields some information about
        which state information is retained.
        params:
            inp: The input state
            projection: The projected output state
        return:
            fig: A figure (matplotlib.pyplot.Figure)
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import scipy as sp
        
        def _annotate_correlation(data, **kws):
            """
            Internal helper-function to annotate correlation text to a 
            figure.
            params:
                data: A pandas dataframe that contains the plotted data (pd.DataFrame)
            return:
                None
            """
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
        p.map_dataframe(_annotate_correlation)
        return p.fig
        


class EncoderCompressor(NetworkCompressor):
    """
    EncoderCompressor is a NetworkCompressor class that compresses the state by funnelling it through a neural network. It has an Encoder structure.
    """
    def train(self, buffer, time_horizon):
        """
        Train the network.
        params:
            buffer: The replay-buffer of the used agent
            time_horizon: The time-horizon out of which the contrasts 
                are sampled
        return:
            loss: The current training loss
        """
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
        """
        Compare input and output of the network regarding retained state-information
        params:
            batch_size: The batch-size used for sampling
            buffer: The replay-buffer of the used agent
        return:
            fig: A figure (matplotlib.pyplot.Figure)
        """
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
    """
    AutoEncoderCompressor is a NetworkCompressor class that compresses the state by funnelling it through a neural network. It has an AutoEncoder structure.
    """
    def train(self, buffer, time_horizon):
        """
        Train the network.
        params:
            buffer: The replay-buffer of the used agent
            time_horizon: Not used, just to enable compatibility with the EncoderCompressor class
        return:
            loss: The current training loss
        """
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
        """
        Compare input and output of the network regarding retained state-information
        params:
            batch_size: The batch-size used for sampling
            buffer: The replay-buffer of the used agent
        return:
            fig: A figure (matplotlib.pyplot.Figure)
        """
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
    """
    torch.nn.Module that represents a simple encoder network.
    """
    def __init__(self, state_dim, subgoal_dim, hidden_layers=[128]):
        """
        Initialize EncoderNetwork class.
        params:
            state_dim: The dimension of the state (int)
            subgoal_dim: The dimension of the subgoal (int)
            hidden_layers: The neurons of each hidden_layer (list(int))
        """
        super(EncoderNetwork, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], subgoal_dim)
                
    def forward(self, state):
        """
        Forward pass of the network.
        params:
            state: The original state as input
        return:
            representation: The encoded state
            projection: A nonlinear projection of the representation
                (in this small network it is just the representation 
                again)
        """
        representation = F.relu(self.l1(state))
        representation = torch.tanh(self.l2(representation))
        # simple structure where representation and projection are the same
        return representation, representation


class AutoEncoderNetwork(nn.Module):
    """
    torch.nn.Module that represents a simple autoencoder network.
    """
    def __init__(self, state_dim, subgoal_dim, hidden_layers=[128]):
        """
        Initialize AutoEncoderNetwork class.
        params:
            state_dim: The dimension of the state (int)
            subgoal_dim: The dimension of the subgoal (int)
            hidden_layers: The neurons of each hidden_layer (list(int))
        """

        super(AutoEncoderNetwork, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], subgoal_dim)
        self.l3 = nn.Linear(subgoal_dim, hidden_layers[0])
        self.l4 = nn.Linear(hidden_layers[0], state_dim)
                
    def forward(self, state):
        """
        Forward pass of the network.
        params:
            state: The original state as input
        return:
            representation: The encoded state
            reconstruction: The decoded representation
        """

        representation = F.relu(self.l1(state))
        representation = torch.tanh(self.l2(representation))
        reconstruction = F.relu(self.l3(representation))
        #projection = torch.tanh(self.l4(projection))
        reconstruction = self.l4(reconstruction)

        return representation, reconstruction



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
