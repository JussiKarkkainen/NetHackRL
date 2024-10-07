import numpy as np
from tinygrad import Tensor

### Implementations of pure functions for different training algorithms


#### PPO 
def compute_returns(rewards, gamma=0.99):
  returns = Tensor.zeros(rewards.shape).contiguous()
  R = Tensor.zeros(rewards.shape[0], 1)
  for t in reversed(range(rewards.shape[1])):
    R = rewards[:, t, :] + gamma * R
    returns[:, t, :] = R
  return returns

@Tensor.train()
def ppo_update(model, optimizer, states_rgb, states_tl, states_bl, actions, 
               prev_actions, returns, advantages, old_log_probs, hiddens, cells, epsilon=0.2):
  
  states_rgb, states_tl, states_bl, prev_actions = states_rgb.squeeze(), states_tl.squeeze(), states_bl.squeeze(), prev_actions.squeeze()

  B, T = states_rgb.shape[0], states_rgb.shape[1]

  # Merge batch and seq_len dims for the encoder
  mb_states_rgb = states_rgb.view(-1, states_rgb.shape[2], states_rgb.shape[3], states_rgb.shape[4])
  mb_tl = states_tl.view(-1, states_tl.shape[2])
  mb_bl = states_bl.view(-1, states_bl.shape[2], states_bl.shape[3])
  mb_prev_actions = prev_actions.view(-1, 1)

  encodings = model.encode(mb_states_rgb, mb_tl, mb_bl, mb_prev_actions)
  encodings = encodings.view(B, T, -1)

  # Get the first hiddens and cell states from each sequence
  mb_h = hiddens[:, 0].view(1, B, hiddens.shape[-1]).squeeze()
  mb_c = cells[:, 0].view(1, B, cells.shape[-1]).squeeze()

  
  log_probs, new_values, _ = model.recurrent(encodings, mb_h, mb_c)

  new_log_probs = log_probs.gather(2, actions.unsqueeze(dim=2)).squeeze()
  ratios = Tensor.exp(new_log_probs - old_log_probs.squeeze())
  surr1 = ratios * advantages
  surr2 = Tensor.clamp(ratios, 1. - epsilon, 1. + epsilon) * advantages

  policy_loss = -Tensor.minimum(surr1, surr2).mean()
  value_loss = returns.sub(new_values).square().mean()
  loss = value_loss + policy_loss

  optimizer.zero_grad()
  loss.realize().backward()
  optimizer.step()
  return loss



#### Behavioral Cloning

def bc_update(model, optimizer, h, c, obs, tl, bl, action_targets, prev_actions):
  action_dists, _, _ = model(obs, tl, bl, prev_actions, h, c)
  action_dists = action_dists.view(action_dists.shape[0]*action_dists.shape[1], -1)
  correct_log_probs = action_dists.gather(dim=1, index=action_targets)
  loss = -correct_log_probs.mean()
  optimizer.zero_grad()
  loss.realize().backward()
  # TODO: Tinygrad gradient clipping
  optimizer.step()
  return loss

