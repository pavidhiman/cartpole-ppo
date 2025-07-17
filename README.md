Training an RL agent with Proximal Policy Optimization (PPO) from scratch using PyTorch (no SB3) to solve the classic CartPole-v1 control problem. 

## Background

**CartPole**
- CartPole is an RL environment where a pole is balanced upright on a cart. The cart will move left and right and the goal is to balance the pole vertically for as long as possible
- Agent receives `+1` reward for each time step the pole remains vertical
- Episode will end if the pole angle is too large, the cart moves out of bounds or the pole survives for 500 steps 

**PPO**
- Classic RL algorithm using exploration 
- The agent collects data overtime through observations, actions and rewards and calculates the advantages. Ultimately, it updates the policy slightly using a clipped objective (allows it to stay close to the old policy to prevent large jumps)

**Actor-Critic Network**
- Actor: decides what to do and outputs an action (policy) (`action_logits = actor(obs)`)
- The actor outputs a probability distribution over actions and learns the best policy to maximize the reward. In PPO it selects actions and updates the policy using the clipped loss
- Critic: judges how good the action was and thus, estimates a value (`value = critic(obs)`)
- The critic outputs a single number/value estimate of the expected future reward from that state. In PPO is calculates the advantage (how much better an action is than the average) 

- Contains: shared based (`torso`), policy head (`actor: outputs actions`) and value head (`critic: estimates state value`)