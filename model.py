import nle
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import os
import argparse
import torch.multiprocessing as mp
from functools import reduce
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2

COLORS = ["#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080",
          "#808080", "#C0C0C0", "#FF0000", "#00FF00", "#FFFF00", "#0000FF", "#FF00FF",
          "#00FFFF", "#FFFFFF"]

def cache_ascii_char(rescale_font_size=9):
  font = ImageFont.truetype("Hack-Regular.ttf", 9)
  dummy_text = "".join([(chr(i) if chr(i).isprintable() else " ") for i in range(256)])
  bboxes = np.array([font.getbbox(char) for char in dummy_text])
  image_width = bboxes[:, 2].max() # 6
  image_height = bboxes[:, 3].max() # 11
  _, _, image_width, image_height = font.getbbox(dummy_text)
  image_width = int(np.ceil(image_width / 256) * 256)

  char_width = rescale_font_size
  char_height = rescale_font_size

  char_array = np.zeros((256, 16, char_height, char_width, 3), dtype=np.uint8)

  image = Image.new("RGB", (image_width, image_height))
  image_draw = ImageDraw.Draw(image)
  for color_index in range(16):
    image_draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0))
    image_draw.text((0, 0), dummy_text, font=font, fill=COLORS[color_index], spacing=0)
    arr = np.array(image).copy()
    arrs = np.array_split(arr, 256, axis=1)
    for char_index in range(256):
      char = arrs[char_index]
      if rescale_font_size:
        char = cv2.resize(char, (rescale_font_size, rescale_font_size), interpolation=cv2.INTER_AREA)
      char_array[char_index, color_index] = char
  return char_array

class CharToImage(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.cache_array = cache_ascii_char()

  def _render_to_image(self, obs):
    chars = obs["tty_chars"][1:-2, :]
    colors = np.clip(obs["tty_colors"], 0, 15)
    
    # 11*chars.shape[0], 6*chars.shape[1] for full image
    # 11 for better looking image
    pixel_obs = np.zeros((9*12, 9*12, 3), dtype=np.float32)

    for i in range(12): # chars.shape[0] for full screen
      for j in range(12): # chars.shape[1] for full screen
        color = colors[i][j]
        char = self.cache_array[chars[i][j]][color]
        pixel_obs[i*9:(i+1)*9, j*9:(j+1)*9, :] = char
  
    obs["rgb_image"] = pixel_obs

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self._render_to_image(obs)
    return obs, reward, done, info

  def reset(self, **kwargs):
    obs = self.env.reset(**kwargs)
    self._render_to_image(obs)
    return obs

class PrevActionsWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.prev_action = 0
    obs_spaces = {"prev_actions": self.env.action_space}
    obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
    self.observation_space = gym.spaces.Dict(obs_spaces)

  def reset(self, **kwargs):
    self.prev_action = 0
    obs = self.env.reset(**kwargs)
    obs["prev_actions"] = np.array([self.prev_action])
    return obs

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    self.prev_action = action
    obs["prev_actions"] = np.array([self.prev_action])
    return obs, reward, done, info

class NetHackEncoder(nn.Module):
  def __init__(self, conv_channels, fc_dims, bl_conv_dims):
    super().__init__()
    convs = []
    in_channels = 3
    kernels_strides = [(8, 6), (4, 2), (3, 2), (3, 1)]
    for layer, (kernel_size, stride) in zip(range(len(conv_channels)), kernels_strides):
      out_channels = conv_channels[layer]
      convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
      if layer != len(conv_channels) - 1:
        convs.append(nn.ELU())
      in_channels = out_channels
    self.screen_convs = nn.Sequential(*convs)
    self.screen_out_fc = nn.Linear(fc_dims["screen_in"], fc_dims["screen_out"])

    self.tl_enc = nn.Sequential(nn.Linear(fc_dims["tl_in"], fc_dims["tl_hidden"]), nn.ELU(), \
                                nn.Linear(fc_dims["tl_hidden"], fc_dims["tl_out"]))

    bl_convs = []
    for in_ch, out_ch, kernel, stride in bl_conv_dims:
      bl_convs.append(nn.Conv1d(in_ch, out_ch, kernel, stride=stride))
      bl_convs.append(nn.ELU())

    self.bl_conv_net = nn.Sequential(*bl_convs)
    self.bl_out_fc = nn.Sequential(nn.Linear(fc_dims["bl_in"], fc_dims["bl_hidden"]), nn.ELU(), \
                                   nn.Linear(fc_dims["bl_hidden"], fc_dims["bl_out"]))

    
  def forward(self, x_rgb, tl, bl, prev_action):
    B = x_rgb.shape[0]
    x_rgb = x_rgb / 255.0
    conv_out = self.screen_convs(x_rgb)
    rgb_flatten = conv_out.reshape(B, -1)
    screen_enc = self.screen_out_fc(rgb_flatten)

    tl = F.one_hot(tl.long(), 256).reshape(B, -1).float()
    tl_enc = self.tl_enc(tl)

    # https://github.com/BartekCupial/sample-factory/blob/master/sf_examples/nethack/models/chaotic_dwarf.py
    bl = bl.view(B, -1)
    chars_normalised = (bl - 32) / 96
    numbers_mask = (bl > 44) * (bl < 58)
    digits_normalised = numbers_mask * (bl - 47) / 10
    bl_norm = torch.stack([chars_normalised, digits_normalised], dim=1)

    bl_enc = self.bl_conv_net(bl_norm).reshape(B, -1)
    bl_enc = self.bl_out_fc(bl_enc)

    pre_action_one_hot = F.one_hot(prev_action, 23).view(B, 23)
    encodings = torch.cat([tl_enc, screen_enc, bl_enc, pre_action_one_hot], dim=-1)
    return encodings

class NetHackModel(nn.Module):
  def __init__(self, config, device):
    super().__init__()
    self.config = config
    self.device = device
    self.encoder = NetHackEncoder(conv_channels=config["conv_channels"], fc_dims=config["fc_dims"], bl_conv_dims=config["bl_conv_dims"])
    self.core = nn.LSTM(input_size=config["lstm_input"], hidden_size=config["lstm_hidden"], batch_first=True)
    self.actor = nn.Linear(config["lstm_hidden"], config["actor_out"])
    self.critic = nn.Linear(config["lstm_hidden"], 1)

  def init_lstm(self):
    return torch.zeros((1, 1, self.config["lstm_hidden"])), torch.zeros((1, 1, self.config["lstm_hidden"]))

  def encode(self, image, tl, bl, prev_action):
    encodings = self.encoder(image, tl, bl, prev_action)
    B, D = encodings.shape
    encodings = encodings.unsqueeze(0)
    return encodings

  def recurrent(self, encodings, h, c):
    o, (h, c) = self.core(encodings, (h.to(self.device), c.to(self.device)))
    action_dist = self.actor(o)
    value = self.critic(o)
    action_prob = F.softmax(action_dist, dim=-1)
    return action_prob, value, (h, c)

  def forward(self, image, tl, bl, prev_action, h, c):
    encodings = self.encode(image, tl, bl, prev_action)
    action_prob, value, (h, c) = self.recurrent(encodings, h, c)
    return action_prob, value, (h, c)

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def data_worker(worker_id, env_conf, model, device, shared_queue, args):
  env = gym.make(args.env)
  env = CharToImage(env)
  env = PrevActionsWrapper(env)

  while True:
    done = False
    state = env.reset()

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

      next_state, reward, done, info = env.step(action.cpu().item())

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

def train(args):
  env_conf = make_config(args)

  if os.getenv("LOG"):
    run = wandb.init(project="PPO-Baseline", config = env_conf)

  model = NetHackModel(score_conf, device)
  model.share_memory()
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=env_conf["lr"])

  shared_queue = mp.Queue()

  workers = []
  for worker_id in range(env_conf["num_workers"]):
    worker = mp.Process(target=data_worker, args=(worker_id, env_conf, model, device, shared_queue, args))
    worker.start()
    workers.append(worker)

  print(f"Training on device {device}")
  print(f"Number of trainable parameters: {count_parameters(model)}")
  
  model.to(device)
  
  for training_step in range(env_conf["training_steps"]):
    st = time.perf_counter()

    all_rollouts = []
    for _ in range(env_conf["num_workers"]):
      rollout = shared_queue.get()
      if rollout: 
        all_rollouts.append(rollout)
        if os.getenv("LOG"):
          wandb.log({"Reward": torch.sum(rollout["rewards"], dim=[0, 1]).item()})

    if not all_rollouts: 
      continue
    
    buffer = {
      key: torch.cat([r[key] for r in all_rollouts], dim=0)
      for key in all_rollouts[0]
    }

    returns = compute_returns(buffer["rewards"])

    advantages = returns - buffer["values"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    loss = ppo_update(model, optimizer, buffer["states_rgb"], buffer["states_tl"], buffer["states_bl"],
               buffer["actions"], buffer["prev_actions"], returns, advantages, buffer["log_probs_old"],
               buffer["hiddens"], buffer["cells"])
    
    if os.getenv("LOG"):
      wandb.log({"Loss": loss})

    et = time.perf_counter()
    print(f"Single training step took: {et - st} seconds")
    print(f"Loss on episode: {training_step} was {loss.item():.4f}")

  for worker in workers:
    worker.terminate()
    
  torch.save(model.state_dict(), env_conf["default_model_path"])
  evaluate_model(env_conf["default_model_path"])

def evaluate_model(model_path):
  env = gym.make(args.env)
  env = CharToImage(env)
  env = PrevActionsWrapper(env)
  model = NetHackModel(score_conf, device)
  model.load_state_dict(torch.load(model_path))
  model.eval()
  avg_rewards = []
  for eval_episode in range(10): # Average of 10 episodes
    done = False
    state = env.reset()
    rewards = []
    h, c = model.init_lstm()
    while not done:
      rgb_image = torch.from_numpy(state["rgb_image"]).unsqueeze(0).permute(0, 3, 1, 2) # B, C, H, W
      tl, bl = torch.from_numpy(state["tty_chars"][0, :]).long().unsqueeze(0), torch.from_numpy(state["tty_chars"][-2:, :]).float().unsqueeze(0)
      prev_action = torch.from_numpy(state["prev_actions"])
      action_dist, valuem, _ = model(rgb_image, tl, bl, prev_action, h, c)
      action_dist = action_dist.squeeze()
      action = torch.multinomial(action_dist, num_samples=1)
      next_state, reward, done, info = env.step(action.item())
      # env.render()
      rewards.append(reward)
      if done:
        break
    episode_reward = sum(rewards)
    print(f"Reward on Eval episode: {eval_episode} was: {episode_reward}")
    avg_rewards.append(episode_reward)
  print(f"Average reward over 10 episodes was: {sum(avg_rewards) / len(avg_rewards)}")

score_conf = {
    "conv_channels": [32, 64, 128, 128],
    "fc_dims": {
      "screen_in": 128,
      "screen_out": 512,
      "tl_in": 20480,
      "tl_hidden": 128,
      "tl_out": 128,
      "bl_in": 2304,
      "bl_hidden": 128,
      "bl_out": 128
    },
    "bl_conv_dims": [[2, 32, 8, 4], [32, 64, 4, 1]],
    "lstm_input": 791,
    "lstm_hidden": 512,
    "actor_out": 23
}

def make_config(args):
  env_conf = {
    "default_model_path": f"checkpoints/run-{time.strftime('%Y%m%d-%H%M%S')}.pt",
    "max_env_steps": 10000,
    "seq_len": 256,
    "lr": 3e-4,
    "num_workers": args.num_data_workers,
    "training_steps": args.training_steps,
    "obs_image_shape": (108, 108),
    "obs_tl_shape": 80,
    "obs_bl_shape": 80,
  }
  return env_conf

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--env", help="Enter the environment, default: NetHackScore-v0", 
                      default="NetHackScore-v0")
  parser.add_argument("--eval", help="Eval a given model, include a path to model file", 
                      action="store_true")
  parser.add_argument("--checkpoint_path", help="Path to a model", default=None)
  parser.add_argument("--training_steps", type=int, help="Number of training steps, default: 200", default=200)
  parser.add_argument("--num_data_workers", type=int, help="Number of data workers, default: 2", default=2)
  args = parser.parse_args()
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  if args.eval:
    if not args.checkpoint_path:
      raise Exception("No model path specified for evaluation")
    model_path = args.checkpoint_path
    evaluate_model(model_path)
    exit()
    
  train(args)
  
