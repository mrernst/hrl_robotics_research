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

        self.model_path = model_path
        # parameters
        self.max_action = max_action
        self.discount = discount
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.tau = tau
        self.name = name

        self.curr_train_metrics = {}
        self.total_it = 0

    def policy(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action = self.actor(torch.cat([state, goal], 1))

        if to_numpy:
            return action.cpu().data.numpy().squeeze()

        return action.squeeze()

    def policy_with_noise(self, state, goal, to_numpy=True):
        state = get_tensor(state)
        goal = get_tensor(goal)
        action = self.actor(torch.cat([state, goal], 1))

        action = action + self._sample_exploration_noise(action)
        action = torch.min(action,  self.actor.max_action)
        action = torch.max(action, -self.actor.max_action)

        if to_numpy:
            return action.cpu().data.numpy().squeeze()

        return action.squeeze()

    def _sample_exploration_noise(self, actions):
        mean = torch.zeros(actions.size()).to(device)
        var = torch.ones(actions.size()).to(device)
        #expl_noise = self.expl_noise - (self.expl_noise/1200) * (self.total_it//10000)
        return torch.normal(mean, self.expl_noise*var)

    def _train(self, state, goal, action, reward, next_state, next_goal, not_done):
        self.total_it += 1

        # check_
        state = torch.cat([state, goal], 1).to(device)
        next_state = torch.cat([next_state, goal], 1).to(device)
        # This is probably not correct!!

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = self.actor_target(next_state) + noise
            next_action = torch.min(next_action,  self.actor.max_action)
            next_action = torch.max(next_action, -self.actor.max_action)
            #.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.smooth_l1_loss(current_Q1, target_Q) + \
            F.smooth_l1_loss(current_Q2, target_Q)

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
        for origin_param, target_param in zip(target.parameters(), origin.parameters()):
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
        name='high',
    ):

        super(HighLevelController, self).__init__(
            state_dim, goal_dim, action_dim, max_action, model_path, actor_lr, critic_lr, actor_hidden_layers, critic_hidden_layers, expl_noise, policy_noise,
            noise_clip, discount, policy_freq, tau, name
        )

        self.action_dim = action_dim
        # needed for off_policy_correction

    def off_policy_corrections(
        self,
        low_con,
        batch_size,
        sgoals,
        states,
        actions,
        candidate_goals=8
    ):

        first_s = [s[0] for s in states]  # First x
        last_s = [s[-1] for s in states]  # Last x

        # Shape: (batch_size, 1, subgoal_dim)
        # diff = 1
        diff_goal = (np.array(last_s) -
                     np.array(first_s))[:, np.newaxis, :self.action_dim]

        # Shape: (batch_size, 1, subgoal_dim)
        # original = 1
        # random = candidate_goals
        original_goal = np.array(sgoals)[:, np.newaxis, :]
        random_goals = np.random.normal(loc=diff_goal, scale=.5*self.max_action[None, None, :],
                                        size=(batch_size, candidate_goals, original_goal.shape[-1]))
        random_goals = random_goals.clip(-self.max_action, self.max_action)

        # Shape: (batch_size, 10, subgoal_dim)
        candidates = np.concatenate(
            [original_goal, diff_goal, random_goals], axis=1)
        #states = np.array(states)[:, :-1, :]
        actions = np.array(actions)
        seq_len = len(states[0])

        # For ease
        new_batch_sz = seq_len * batch_size
        action_dim = actions[0][0].shape
        obs_dim = states[0][0].shape
        ncands = candidates.shape[1]

        true_actions = actions.reshape((new_batch_sz,) + action_dim)
        observations = states.reshape((new_batch_sz,) + obs_dim)
        goal_shape = (new_batch_sz, self.action_dim)
        # observations = get_obs_tensor(observations, sg_corrections=True)

        # batched_candidates = np.tile(candidates, [seq_len, 1, 1])
        # batched_candidates = batched_candidates.transpose(1, 0, 2)

        policy_actions = np.zeros((ncands, new_batch_sz) + action_dim)

        for c in range(ncands):
            subgoal = candidates[:, c]
            candidate = (subgoal + states[:, 0, :self.action_dim]
                         )[:, None] - states[:, :, :self.action_dim]
            candidate = candidate.reshape(*goal_shape)
            policy_actions[c] = low_con.policy(observations, candidate)

        difference = (policy_actions - true_actions)
        difference = np.where(difference != -np.inf, difference, 0)
        difference = difference.reshape(
            (ncands, batch_size, seq_len) + action_dim).transpose(1, 0, 2, 3)

        logprob = -0.5*np.sum(np.linalg.norm(difference, axis=-1)**2, axis=-1)
        max_indices = np.argmax(logprob, axis=-1)

        return candidates[np.arange(batch_size), max_indices]

    def train(self, replay_buffer, low_con):
        states, goals, actions, n_states, rewards, not_done, states_arr, actions_arr = replay_buffer.sample()

        actions = self.off_policy_corrections(
            low_con,
            replay_buffer.batch_size,
            actions.cpu().data.numpy(),
            states_arr.cpu().data.numpy(),
            actions_arr.cpu().data.numpy())

        actions = get_tensor(actions)
        return self._train(states, goals, actions, rewards, n_states, goals, not_done)

    pass


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
        name='low',
    ):

        super(LowLevelController, self).__init__(
            state_dim, goal_dim, action_dim, max_action, model_path,
            actor_lr, critic_lr, actor_hidden_layers, critic_hidden_layers, expl_noise, policy_noise,
            noise_clip, discount, policy_freq, tau
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
