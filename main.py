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
    
# rollout buffer - temp storage of data until next model update 
class RolloutBuffer:
    def __init(self):
        self.clear()
    
    def store(self, state, action, reward, done, logp, value): # storing 1 frame into buffer
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logps.append(logp)
        self.values.append(value)
    
    def clear(self):
        self.states, self.actions  = [], []
        self.rewards, self.dones   = [], []
        self.logps, self.values    = [], []
        
# GAE 
def compute_gae(rewards, dones, values, next_value):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))): 
        delta = rewards[i] + GAMMA * (1 - dones[i]) * next_value - values[i] # temporal difference - actual vs. predicted
        gae   = delta + GAMMA * LAMBDA * (1 - dones[i]) * gae
        advantages.insert(0, gae)
        next_value = values[i]
    returns = [a + v for a, v in zip(advantages, values)] # ret = adv + estimated value

    # convert to pytorch sensors 
    adv   = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret   = torch.tensor(returns,    dtype=torch.float32, device=device)
    return adv, ret # ret is target for critic network 

def ppo_update(model, optimizer, buffer):
    states  = torch.tensor(buffer.states, dtype=torch.float32, device=device)
    actions = torch.tensor(buffer.actions, dtype=torch.int64,  device=device)
    old_logps = torch.tensor(buffer.logps, dtype=torch.float32, device=device)
    old_values = torch.tensor(buffer.values, dtype=torch.float32, device=device)

    # value of last state for GAE
    with torch.no_grad():
        _, next_value = model(states[-1].unsqueeze(0))
        next_value = next_value.squeeze().item() # convert to int and remove extra dim from unsqueeze 

    advantages, returns = compute_gae(
        buffer.rewards, buffer.dones, old_values.cpu().numpy(), next_value
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # normalize

    # epoch loop 
    dataset_size = states.size(0)
    for _ in range(UPDATE_EPOCHS):
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        for start in range(0, dataset_size, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            batch_states   = states[batch_idx]
            batch_actions  = actions[batch_idx]
            batch_oldlogp  = old_logps[batch_idx]
            batch_returns  = returns[batch_idx]
            batch_adv      = advantages[batch_idx]

            logits, values = model(batch_states)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            # PPO clipped objective 
            ratio = (logp - batch_oldlogp).exp()
            surr1 = ratio * batch_adv # normal update to policy 
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_adv # clipped update to policy 
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss  = ((batch_returns - values.squeeze()) ** 2).mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
def train():
    env = gym.make(ENV_ID)
    env.action_space.seed(SEED)
    obs_dim = env.observation_space.shape[0] # obs_dim = 4: [position, velocity, angle, angular velocity]
    act_dim = env.action_space.n # act_dim = 2: left and right 

    model = ActorCritic(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    buffer = RolloutBuffer()
    episode_rewards, reward_history = [], []

    obs, _ = env.reset(seed=SEED)
    for step in range(1, TOTAL_STEPS + 1):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        logits, value = model(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
        logp   = dist.log_prob(torch.tensor(action, device=device)).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.store(obs, action, reward, done, logp, value.item())

        obs = next_obs
        episode_rewards.append(reward)

        if done: # reset env if done 
            obs, _ = env.reset()
            reward_history.append(sum(episode_rewards))
            episode_rewards.clear()

        # every 2048 (rollout_len) - run training update 
        if step % ROLLOUT_LEN == 0:
            ppo_update(model, optimizer, buffer)
            buffer.clear()

            if reward_history:
                print(f"Step {step:6d}  |  Avg reward (last 5 eps): {np.mean(reward_history[-5:]):.1f}")

    env.close()
    torch.save(model.state_dict(), "ppo_cartpole.pt")
    
    # plot
    plt.plot(reward_history)
    plt.title("Episode reward")
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.savefig("learning_curve.png")
    plt.show()

if __name__ == "__main__":
    train()