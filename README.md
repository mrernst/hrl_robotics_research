# Robotic Grasping
Machine learning playground using OpenAI Gym and PyBullet for robotic grasping.

## Policy Based Methods
### Deep Q Learning (DQN)
### Proximal Policy Optimization (PPO)
### Interactive RL

### Model update using PPO/GAE
The hyperparameters used during training are:

Parameter | Value | Description
------------ | ------------- | -------------
Number of Agents | 1 | Number of agents trained simultaneously
tmax | 20 | Maximum number of steps per episode
Epochs | 10 | Number of training epoch per batch sampling
Batch size | 128 | Size of batch taken from the accumulated  trajectories
Discount (gamma) | 0.993 | Discount rate 
Epsilon | 0.07 | Ratio used to clip r = new_probs/old_probs during training
Gradient clip | 10.0 | Maximum gradient norm 
Beta | 0.01 | Entropy coefficient 
Tau | 0.95 | tau coefficient in GAE
Learning rate | 2e-4 | Learning rate 
Optimizer | Adam | Optimization method