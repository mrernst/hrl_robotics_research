#!/usr/bin/python
# _____________________________________________________________________________

# Modified Author implementation
# Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helper functions
# -----


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def get_tensor(z):
    if len(z.shape) == 1:
        return torch.FloatTensor(z).unsqueeze(0).to(device)
    else:
        return torch.FloatTensor(z).to(device)
        
# net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
# net.apply(init_weights)

# custom classes
# -----


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_layers):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l3 = nn.Linear(hidden_layers[1], action_dim)

        self.max_action = get_tensor(max_action)

    def forward(self, state):
        #print(self.max_action)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_layers[0])
        self.l2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l3 = nn.Linear(hidden_layers[1], 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_layers[0])
        self.l5 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.l6 = nn.Linear(hidden_layers[1], 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Controller(object):
    def __init__(
            self,
            state_dim,
            goal_dim,
            action_dim,
            max_action,
            model_path,
            actor_lr=0.0003,
            critic_lr=0.0003,
            actor_hidden_layers=[256, 256],
            critic_hidden_layers=[256, 256],
            expl_noise=0.1,
            policy_noise=0.2,
            noise_clip=0.5,
            discount=0.99,
            policy_freq=2,
            tau=0.005,
            name='default',
    ):

        self.actor = Actor(state_dim + goal_dim, action_dim,
                           max_action, actor_hidden_layers).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim + goal_dim, action_dim,
                             critic_hidden_layers).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)
        # Huber loss does not punish a noisy large gradient.
        self.critic_loss_fn = F.huber_loss #F.mse_loss

        self.model_path = model_path
        # parameters
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.expl_noise = expl_noise
        self.name = name

        self.curr_train_metrics = {}
        self.total_it = 0

    def policy(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action = self.actor(torch.cat([state, goal], 1))
        if to_numpy:
            return action.cpu().data.numpy().flatten()

        return action.flatten()

    def policy_with_noise(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action = self.actor(torch.cat([state, goal], 1))

        action = action + self._sample_exploration_noise(action)
        action = torch.min(action,  self.actor.max_action)
        action = torch.max(action, -self.actor.max_action)

        if to_numpy:
            return action.cpu().data.numpy().flatten()

        return action.flatten()

    def _sample_exploration_noise(self, actions):
        mean = torch.zeros(actions.size()).to(device)
        var = torch.ones(actions.size()).to(device)
        #expl_noise = self.expl_noise - (self.expl_noise/1200) * (self.total_it//10000)
        #self.actor.max_action*self.expl_noise*var
        return torch.normal(mean, self.expl_noise*var)

    def _train(self, state, goal, action, reward, next_state, next_goal, not_done):
        self.total_it += 1

        # check_??
        state = torch.cat([state, goal], 1).to(device)
        next_state = torch.cat([next_state, next_goal], 1).to(device)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.actor.max_action, self.actor.max_action)
            

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss =  self.critic_loss_fn(current_Q1, target_Q) + \
           self.critic_loss_fn(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Collect metrics
        with torch.no_grad():
            norm_type = 2
            cr_gr_norm = torch.norm(torch.stack([torch.norm(
                p.grad.detach(), norm_type) for p in self.critic.parameters()]), norm_type)
            td_error = (target_Q - current_Q1).mean().cpu().data.numpy()
            self.curr_train_metrics['critic/gradients/norm'] = cr_gr_norm
            self.curr_train_metrics['critic/loss'] = critic_loss
            self.curr_train_metrics['td_error'] = td_error
            self.curr_train_metrics['reward'] = reward.mean()

        #         # tensorboard_writer.add_histogram(f'{self.name}/value/critic_weights',(torch.cat([p.flatten() for p in self.critic.parameters()])).detach().numpy(), self.total_it)

        # Delayed policy updates
        if (self.total_it % self.policy_freq == 0):

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Collect metrics
            with torch.no_grad():
                norm_type = 2
                ac_gr_norm = torch.norm(torch.stack([torch.norm(
                    p.grad.detach(), norm_type) for p in self.actor.parameters()]), norm_type)
                self.curr_train_metrics['actor/gradients/norm'] = ac_gr_norm
                self.curr_train_metrics['actor/loss'] = actor_loss

            # Update the frozen target models
            self._update_target_network(
                self.critic_target, self.critic, self.tau)
            self._update_target_network(
                self.actor_target, self.actor, self.tau)
                
            return {'actor_loss_'+self.name: actor_loss, 'critic_loss_'+self.name: critic_loss}, \
            {'td_error_'+self.name: td_error}

        return {'critic_loss_'+self.name: critic_loss}, \
            {'td_error_'+self.name: td_error}

    def train(self, replay_buffer):
        state, goal, action, next_state, reward, not_done = replay_buffer.sample()
        return self._train(state, goal, action, reward, next_state, goal, not_done)

    def _update_target_network(self, target, origin, tau):
        for origin_param, target_param in zip(origin.parameters(), target.parameters()):
            target_param.data.copy_(
                tau * origin_param.data + (1.0 - tau) * target_param.data)

    def save(self, episode):
        # create episode directory. (e.g. model/2000)
        model_path = os.path.join(self.model_path, str(episode))
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(self.critic.state_dict(), os.path.join(
            model_path, self.name+"_critic"))
        torch.save(self.critic_optimizer.state_dict(),
                   os.path.join(model_path, self.name+"_critic_optimizer"))

        torch.save(self.actor.state_dict(), os.path.join(
            model_path, self.name+"_actor"))
        torch.save(self.actor_optimizer.state_dict(),
                   os.path.join(model_path, self.name+"_actor_optimizer"))

    def load(self, episode):
        # episode is -1, then read most updated
        if episode<0:
            episode_list = map(int, os.listdir(self.model_path))
            episode = max(episode_list)
            print(" " * 80 + "\r" + f'[INFO] Loaded model at episode {episode}')


        model_path = os.path.join(self.model_path, str(episode))

        self.critic.load_state_dict(torch.load(os.path.join(
            model_path, self.name+"_critic"), map_location=torch.device(device)))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(model_path, self.name+"_critic_optimizer"), map_location=torch.device(device)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(
            model_path, self.name+"_actor"), map_location=torch.device(device)))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(model_path, self.name+"_actor_optimizer"), map_location=torch.device(device)))
        self.actor_target = copy.deepcopy(self.actor)


class HighLevelController(TD3Controller):
    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        max_action,
        model_path,
        actor_lr=0.0003,
        critic_lr=0.0003,
        actor_hidden_layers=[256, 256],
        critic_hidden_layers=[256, 256],
        expl_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        discount=0.99,
        policy_freq=2,
        tau=0.005,
        name='meta',
    ):

        super(HighLevelController, self).__init__(
            state_dim, goal_dim, action_dim, max_action, model_path, actor_lr, critic_lr, actor_hidden_layers, critic_hidden_layers, expl_noise, policy_noise,
            noise_clip, discount, policy_freq, tau, name
        )

        self.action_dim = action_dim
        # needed for off_policy_correction

    
    def _off_policy_corrections(
        self, 
        low_con,
        batch_size,
        states,
        actions,
        next_states,
        low_states,
        low_actions,
        candidate_goals=8):
    
        # Scale
        goal_dim = self.action_dim
        spec_range = low_con.max_action.cpu().data.numpy()
        # Sample from normal distribution
        loc = (next_states - states)[:, np.newaxis, :goal_dim]
        # scale = 0.5 * self.max_action[None, None, :]
        scale = 0.25 * self.max_action #c.f. HIRO Paper
        original_goal = np.array(actions[:, np.newaxis, :])
        random_goals = np.random.normal(loc=loc, scale=scale, size=(batch_size, candidate_goals, original_goal.shape[-1]))
        candidates = np.concatenate([original_goal, loc, random_goals], axis=1)
        candidates = candidates.clip(-self.max_action.cpu().data.numpy(), self.max_action.cpu().data.numpy())

        # For ease
        low_actions = np.array(low_actions)
        seq_len = len(low_states[0])
        new_batch_sz = seq_len * batch_size
        low_action_dim = low_actions[0][0].shape
        low_obs_dim = low_states[0][0].shape
        ncands = candidates.shape[1]
        
        true_low_actions = low_actions.reshape((new_batch_sz,) + low_action_dim)
        observations = low_states.reshape((new_batch_sz,) + low_obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        
        pred_actions = np.zeros((ncands, new_batch_sz) + low_action_dim)
        with torch.no_grad():
            for c in range(ncands):
                subgoal = candidates[:, c]
                candidate = (subgoal + low_states[:, 0, :self.action_dim])[:, None] - low_states[:, :, :self.action_dim] #?
                candidate = candidate.reshape(*goal_shape)
                #print(pred_actions.shape[1:])
                pred_actions[c] = low_con.policy(torch.tensor(observations).float(), torch.tensor(candidate).float()).reshape(pred_actions.shape[1:])
        
        difference = (pred_actions - true_low_actions)
        # difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape((ncands, batch_size, seq_len) + low_action_dim).transpose(1, 0, 2, 3)
        
        normalized_error = - np.square(difference) / np.square(spec_range)
        #print(normalized_error.shape)
        fitness = np.sum(normalized_error, axis=(2, 3))
        # - 0.5 * tf.reduce_sum(tf.square(tf.norm(diffn, axis=2)), axis=1) ??
        best_actions = np.argmax(fitness, axis=-1)
        
        return candidates[np.arange(batch_size), best_actions]


    def train(self, replay_buffer, low_con):
        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = replay_buffer.sample()
        
        
        actions = self._off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            states.cpu().data.numpy(),
            actions.cpu().data.numpy(),
            n_states.cpu().data.numpy(),
            states_arr.cpu().data.numpy(),
            actions_arr.cpu().data.numpy())

        actions = get_tensor(actions)
        return self._train(states, goals, actions, rewards, n_states, goals, not_done)

    pass

# 
# 
# def off_policy_correction_tf(subgoal_ranges, target_dim, pi, goal_b, state_b, next_state_b, no_candidates, c_step, state_seq,
#                           action_seq, zero_obs):
#     # TODO Update docstring to real dimensions
#     '''Computes the off-policy correction for the meta-agent.
#     c = candidate_nbr; t = time; i = vector coordinate (e.g. action 0 of 8 dims) 
#     Dim(candidates) = [b_size, g_dim, no_candidates+2 ]
#     Dim(state_b)     = [b_size, state_dim]
#     Dim(action_seq)    = []
#     Dim(prop_goals) = [c_step, b_size, g_dim, no_candidates]'''
#     b_size = state_b.shape[0] # Batch Size
#     g_dim = goal_b[0].shape[0] # Subgoal dimension
#     action_dim = action_seq.shape[-1] # action dimension
#     # States contains state+target. Need to take out only the state.
#     state_seq = state_seq[:,:, :-target_dim]
#     state_b = state_b[:, :-target_dim]
#     next_state_b = next_state_b[:, :-target_dim]
#     # Get the sampled candidates
#     candidates =  _get_goal_candidates(b_size, g_dim, no_candidates, subgoal_ranges, state_b, next_state_b, goal_b)
#     # Take all the possible candidates and propagate them through time using h = st + gt - st+1 cf. HIRO Paper
#     prop_goals = _multi_goal_transition(state_seq, candidates, c_step)
#     # Zero out xy for sub agent, AFTER goals have been calculated from it.
#     state_seq = tf.reshape(state_seq, [b_size * c_step, state_seq.shape[-1]])
#     if zero_obs:
#         state_seq *= tf.concat([tf.zeros([state_seq.shape[0], zero_obs]), tf.ones([state_seq.shape[0], state_seq.shape[1] - zero_obs])], axis=1)
#     best_c = _get_best_c(b_size, c_step, action_dim, g_dim, no_candidates, action_seq, state_seq, prop_goals, pi) 
#     return _get_corrected_goal(b_size, candidates, best_c) 
#     
# def _multi_goal_transition(state_seq, candidates, c_step):
#     # Realize that the multi timestep goal transition can be vectorized to a single calculation.
#     b_size = candidates.shape[0]
#     g_dim = candidates.shape[1]
#     c_step = state_seq.shape[1] 
#     no_candidates = candidates.shape[2]
#     # In state_seq, equal batches are separated by c timesteps if we query a subogoal every c_steps 
#     prop_goals = tf.broadcast_to(tf.expand_dims(candidates, axis=1), [b_size, c_step, g_dim, no_candidates])
#     tmp = tf.broadcast_to(tf.expand_dims(state_seq[:,0,:g_dim], axis=1), [b_size, c_step, g_dim]) - state_seq[:, :, :g_dim] 
#     prop_goals += tf.broadcast_to(tf.expand_dims(tmp, axis=3), [b_size, c_step, g_dim, no_candidates]) 
#     return prop_goals 
#     
# def _get_goal_candidates(b_size, g_dim, no_candidates, subgoal_ranges, state_b, next_state_b, goal_b):
#     # Original Goal
#     orig_goal = tf.expand_dims(goal_b[:, :g_dim], axis=2)
#     # Goal between the states s_t+1 - s_t
#     diff_goal = tf.expand_dims(next_state_b[:, :g_dim] - state_b[:, :g_dim], axis=2)
#     goal_mean = tf.broadcast_to(diff_goal, [b_size, g_dim, no_candidates])
#     # Broadcast the subgoal_ranges to [b_size, g_dim, no_candidates] for use as clipping and as std
#     clip_tensor = tf.expand_dims(tf.broadcast_to(subgoal_ranges, [b_size, subgoal_ranges.shape[0]]), axis=2)
#     clip_tensor = tf.broadcast_to(clip_tensor, [b_size, subgoal_ranges.shape[0], no_candidates+2])
#     goal_std = 0.25 * clip_tensor # cf. HIRO Paper
#     candidates = tf.random.normal([b_size, g_dim, no_candidates], goal_mean, goal_std[:, :, :no_candidates])
#     candidates = tf.concat([orig_goal, diff_goal, candidates], axis=2)
#     candidates = tf.clip_by_value(candidates, -clip_tensor, clip_tensor)
#     return candidates
# 
# def _get_corrected_goal(b_size, candidates, best_c): 
#     corrected_goals = tf.TensorArray(dtype=tf.float32, size=b_size)
#     for b in tf.range(b_size):
#         corrected_goals = corrected_goals.write(b, candidates[b, :, best_c[b]])
#     return corrected_goals.stack() 
#     
# def _get_best_c(b_size, c_step, action_dim, g_dim, no_candidates, action_seq, state_seq, prop_goals, pi):
#     # Compute the logpropabilities for different subgoal candidates
#     max_logprob = tf.constant(-1e9, shape=[b_size,])
#     best_c = tf.zeros([b_size,], dtype=tf.int32) # In Graph mode, elements need to be defined BEFORE the loop
#     action_seq = tf.reshape(action_seq, [b_size * c_step, action_dim])
#     for c in tf.range(no_candidates):
#         goals = prop_goals[:, :, :, c]  # Pick one candidate 
#         # Change dimension of goals so that it can be computed in one batch in the network [b_size * c_step, g_dim]
#         goals = tf.reshape(goals, [b_size * c_step, g_dim])
#         state_action = tf.concat([state_seq, goals], axis=1)
#         pred_action =  pi(state_action)
#         diff = action_seq - pred_action
#         # Have padded the sequences where len(seq)<c_step to the max(c_step) with np.inf
#         # This results in nan in the computed action differences. These are now set to 0
#         # as 0 does not influence the sum over all actions that happens later.
#         diff = tf.where(tf.math.is_nan(diff), 0., diff)
#         # This reshape only works when rehaping consecutive dimension (e.g. [2560,8] to [256,10,8] or vice versa. 
#         # Else elements will be thrown around.
#         diffn = tf.reshape(diff, [b_size, c_step, action_dim])
#         # cf. HIRO Paper
#         logprob = - 0.5 * tf.reduce_sum(tf.square(tf.norm(diffn, axis=2)), axis=1)
#         best_c = tf.where(logprob > max_logprob, c, best_c)
#         max_logprob = tf.where(logprob > max_logprob, logprob, max_logprob)
#     return best_c

class LowLevelController(TD3Controller):

    def __init__(
        self,
        state_dim,
        goal_dim,
        action_dim,
        max_action,
        model_path,
        actor_lr=0.0003,
        critic_lr=0.0003,
        actor_hidden_layers=[256, 256],
        critic_hidden_layers=[256, 256],
        expl_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        discount=0.99,
        policy_freq=2,
        tau=0.005,
        name='sub',
    ):

        super(LowLevelController, self).__init__(
            state_dim, goal_dim, action_dim, max_action, model_path,
            actor_lr, critic_lr, actor_hidden_layers, critic_hidden_layers, expl_noise, policy_noise,
            noise_clip, discount, policy_freq, tau, name
        )

    def train(self, replay_buffer):
        # if not self._initialized:
        #     self._initialize_target_networks()

        states, sgoals, actions, n_states, n_sgoals, rewards, not_done = replay_buffer.sample()

        return self._train(states, sgoals, actions, rewards, n_states, n_sgoals, not_done)


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
