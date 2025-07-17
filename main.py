import gymnasium as gym
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
# hyperparams 
ENV_ID = "CartPole-v1"      
TOTAL_STEPS = 100_000       # total num of env steps before training stops 
ROLLOUT_LEN = 2048          # num of steps collected before each policy update (ie, rollout/batch) - 2048 = power of 2, standard in PPO
UPDATE_EPOCHS = 10          # num of times to reuse same rollout batch to update network
GAMMA = 0.99                # discount factor for future rewards - care 99% about future rewards     
LAMBDA = 0.95               # smoothing factor for Generalized Advantage Estimation (GAE) - (how good the action is - how good I thought it would be)
CLIP_EPS = 0.2              # clip param - limits how much policy can change per step    
LR = 3e-4                   # adam optimizer learning rate (ie, how big are updates to models weight)
ENTROPY_COEF = 0.0          # encourages exploration     
VALUE_COEF = 0.5             
BATCH_SIZE = 64             # size of mini batc during gradient descent per rollout (2048/32 = 64) 
SEED = 42                   # random num generators 

# actor-critic network
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        hidden = 64
        self.torso = nn.Sequential ( # shared feature extractor 
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        self.policy_head = nn.Linear(hidden, act_dim) # actor
        self.value_head = nn.Linear(hidden, 1) # critic 
    
    def forward(self, x):
        x = self.torso(x)
        return self.policy_head(x), self.value_head(x)