import nle
import gymnasium as gym
import numpy as np
from tinygrad import Tensor, nn
from preprocessing import CharToImage, PrevActionsWrapper, normalize_image
from models import NetHackModel


@Tensor.test()
def data_worker(env_conf, score_conf, data_queue):
  env = gym.make(env_conf["env_name"], character=env_conf["character"])
  env = CharToImage(env, env_conf)
  env = PrevActionsWrapper(env)

  while True:
    done = False
    state, info = env.reset()
    
    model = NetHackModel(score_conf)
    nn.state.load_state_dict(model, nn.state.safe_load(env_conf["model_storage"]))

    h, c = model.init_lstm()

    buffer = {"states_rgb": [], "states_tl": [], "states_bl": [], "actions": [], 
              "prev_actions": [], "rewards": [], "values": [], "log_probs_old": [],
              "hiddens": [], "cells": []}

    step = 0
    while not done and step < 48:# env_conf["max_env_steps"]:
      rgb_image = normalize_image(Tensor(state["rgb_image"]).unsqueeze(0)).permute(0, 3, 1, 2) # B, C, H, W
      tl, bl = Tensor(state["tty_chars"][0, :]).unsqueeze(0), Tensor(state["tty_chars"][-2:, :]).float().unsqueeze(0)
      prev_action = Tensor(state["prev_actions"]).unsqueeze(dim=0)

      log_probs, value, (h, c) = model(rgb_image, tl, bl, prev_action, h, c)
      log_probs_s = log_probs.squeeze()
      u = Tensor.uniform(shape=log_probs_s.shape)
      action = Tensor.argmax(log_probs_s - Tensor.log(-Tensor.log(u)), axis=-1)

      next_state, reward, terminated, truncated, info = env.step(action.realize().item())
      done = terminated or truncated

      buffer["states_rgb"].append(rgb_image)
      buffer["states_tl"].append(tl)
      buffer["states_bl"].append(bl)
      buffer["actions"].append(action)
      buffer["prev_actions"].append(prev_action)
      buffer["rewards"].append(Tensor([reward]))
      buffer["values"].append(value)
      buffer["log_probs_old"].append(log_probs[:, :, action])
      buffer["hiddens"].append(h)
      buffer["cells"].append(c)

      state = next_state
      step += 1
    
    buffer = {key: Tensor.stack(*buffer[key], dim=0) for key in buffer}
    
    buffer = {key: pad_sequence(buffer[key].split(env_conf["seq_len"])).detach().realize() \
        for key in buffer} if step > env_conf["seq_len"] else None # NOTE: We ignore rollouts that are shorter than seq_len, should they be included?
    
    data_queue.put(buffer)


def pad_sequence(sequences, batch_first=True, padding_value=0.0):
  max_len = max([seq.shape[0] for seq in sequences])
  padded_seqs = []
  for seq in sequences:
    padding = Tensor.full((max_len - seq.shape[0], *seq.shape[1:]), fill_value=padding_value)
    padded_seq = seq.cat(padding, dim=0)
    assert padded_seq.shape[0] == max_len
    padded_seqs.append(padded_seq)
  return Tensor.stack(*padded_seqs) if batch_first else Tensor.stack(*padded_seqs).transpose(0, 1)
