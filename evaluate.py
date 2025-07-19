import torch
import gymnasium as gym
from main import ActorCritic, device

env = gym.make("CartPole-v1", render_mode="human")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

model = ActorCritic(obs_dim, act_dim).to(device)
model.load_state_dict(torch.load("ppo_cartpole.pt"))
model.eval()

obs, _ = env.reset()
done = False

while True:
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits, value = model(obs_tensor)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample().item()

    # logs for action
    print(f"Obs: {obs}") # raw obs: (position, velocity, pole angle, pole velocity)
    print(f"Action: {action}") # 0 = left, 1 = right
    print(f"Estimated Value: {value.item():.2f}") # critics estimate
    print("=" * 30)
    
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    if done:
        obs, _ = env.reset()
