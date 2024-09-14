import nle
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import CharToImage, PrevActionsWrapper


def data_worker(worker_id, env_conf, model, device, shared_queue, args):
  env = gym.make(args.env)
  env = CharToImage(env, env_conf)
  env = PrevActionsWrapper(env)

  while True:
    done = False
    state, info = env.reset()

    h, c = model.init_lstm()

    buffer = {"states_rgb": [], "states_tl": [], "states_bl": [], "actions": [], 
              "prev_actions": [], "rewards": [], "values": [], "log_probs_old": [],
              "hiddens": [], "cells": []}

    step = 0
    while not done and step < env_conf["max_env_steps"]:
      rgb_image = torch.from_numpy(state["rgb_image"]).unsqueeze(0).permute(0, 3, 1, 2).to(device) # B, C, H, W
      tl, bl = torch.from_numpy(state["tty_chars"][0, :]).long().unsqueeze(0).to(device), torch.from_numpy(state["tty_chars"][-2:, :]).float().unsqueeze(0).to(device)
      prev_action = torch.from_numpy(state["prev_actions"]).unsqueeze(dim=0).to(device)

      action_dist, value, (h, c) = model(rgb_image, tl, bl, prev_action, h, c)
      action = torch.multinomial(action_dist.squeeze(), num_samples=1).to(device)

      next_state, reward, terminated, truncated, info = env.step(action.cpu().item())
      done = terminated or truncated

      buffer["states_rgb"].append(rgb_image)
      buffer["states_tl"].append(tl)
      buffer["states_bl"].append(bl)
      buffer["actions"].append(action)
      buffer["prev_actions"].append(prev_action)
      buffer["rewards"].append(torch.tensor([reward], device=device))
      buffer["values"].append(value)
      buffer["log_probs_old"].append(torch.log(action_dist[:, :, action]))
      buffer["hiddens"].append(h)
      buffer["cells"].append(c)

      state = next_state
      step += 1

    buffer = {key: torch.cat(buffer[key], dim=0).detach() for key in buffer}
    
    buffer = {key: nn.utils.rnn.pad_sequence(buffer[key].split(env_conf["seq_len"]), batch_first=True) \
        for key in buffer} if step > env_conf["seq_len"] else None # NOTE: We ignore rollouts that are shorter than seq_len=256, should they be included?
    shared_queue.put(buffer)
