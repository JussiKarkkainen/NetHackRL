import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


### Implementations of pure functions for different training algorithms


#### PPO 
def compute_returns(rewards, gamma=0.99):
  returns = torch.zeros(rewards.shape)
  R = torch.zeros(rewards.shape[0], 1)
  for t in reversed(range(rewards.shape[1])):
    R = rewards[:, t].unsqueeze(dim=1) + gamma * R
    returns[:, t] = R.squeeze()
  return returns

def ppo_update(model, optimizer, states_rgb, states_tl, states_bl, actions, 
               prev_actions, returns, advantages, old_log_probs, hiddens, cells, epsilon=0.2):
  
  B, T = states_rgb.shape[0], states_rgb.shape[1]

  # Merge batch and seq_len dims for the encoder
  mb_states_rgb = states_rgb.view(-1, states_rgb.shape[2], states_rgb.shape[3], states_rgb.shape[4])
  mb_tl = states_tl.view(-1, states_tl.shape[2])
  mb_bl = states_bl.view(-1, states_bl.shape[2], states_bl.shape[3])
  mb_prev_actions = prev_actions.view(-1, 1)

  encodings = model.encode(mb_states_rgb, mb_tl, mb_bl, mb_prev_actions)
  encodings = encodings.view(B, T, -1)

  # Get the first hiddens and cell states from each sequence
  mb_h = hiddens[:, 0].view(1, B, hiddens.shape[-1])
  mb_c = cells[:, 0].view(1, B, cells.shape[-1])
  
  new_action_dists, new_values, _ = model.recurrent(encodings, mb_h, mb_c)

  new_log_probs = torch.log(new_action_dists.gather(2, actions.unsqueeze(dim=2).long())).squeeze()
  ratios = torch.exp(new_log_probs - old_log_probs)
  surr1 = ratios * advantages
  surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages

  policy_loss = -torch.min(surr1, surr2).mean()
  value_loss = F.mse_loss(returns, new_values.squeeze())
  loss = value_loss + policy_loss

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss



#### Behavioral Cloning

def behavioural_cloning():
  pass

