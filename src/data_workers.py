import nle
import gymnasium as gym
import numpy as np
from tinygrad import Tensor
from preprocessing import CharToImage, PrevActionsWrapper
from models import NetHackModel


def data_worker(env_conf, model, shared_queue):
  env = gym.make(env_conf["env_name"], character=env_conf["character"])
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
      rgb_image = Tensor(state["rgb_image"]).unsqueeze(0).permute(0, 3, 1, 2) # B, C, H, W
      tl, bl = Tensor(state["tty_chars"][0, :]).unsqueeze(0), Tensor(state["tty_chars"][-2:, :]).float().unsqueeze(0)
      prev_action = Tensor(state["prev_actions"]).unsqueeze(dim=0)

      action_dist, value, (h, c) = model(rgb_image, tl, bl, prev_action, h, c)
      action = action_dist.squeeze().multinomial(num_samples=1)

      next_state, reward, terminated, truncated, info = env.step(action.item())
      raise Exception
      done = terminated or truncated

      buffer["states_rgb"].append(rgb_image)
      buffer["states_tl"].append(tl)
      buffer["states_bl"].append(bl)
      buffer["actions"].append(action)
      buffer["prev_actions"].append(prev_action)
      buffer["rewards"].append(Tensor([reward]))
      buffer["values"].append(value)
      buffer["log_probs_old"].append(Tensor.log(action_dist[:, :, action]))
      buffer["hiddens"].append(h)
      buffer["cells"].append(c)

      state = next_state
      step += 1

    buffer = {key: Tensor.cat(buffer[key], dim=0).detach() for key in buffer}
    
    buffer = {key: pad_sequence(buffer[key].split(env_conf["seq_len"])) \
        for key in buffer} if step > env_conf["seq_len"] else None # NOTE: We ignore rollouts that are shorter than seq_len, should they be included?

    shared_queue.put(buffer)


def pad_sequence(sequences, batch_first=True, padding_value=0.0):
  max_len = max([len(seq) for seq in sequences])

  padded_seqs = []
  for seq in sequences:
    padding = [padding_value] * (max_len - len(seq))
    padded_seq = seq + padding
    padded_seqs.append(padded_seq)
  
  return Tensor.stack(padded_seqs) if batch_first else Tensor.stack(padded_seqs).transpose(0, 1)
