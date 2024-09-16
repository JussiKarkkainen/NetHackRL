import nle
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
import os
import torch.multiprocessing as mp
from functools import reduce
from preprocessing import CharToImage, PrevActionsWrapper, cache_ascii_char, preprocess_dataset
from nld_aa_pretraining import dataset
from models import NetHackModel
from configs import make_configs
from data_workers import data_worker
from algs import ppo_update, compute_returns

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model_path):
  env_conf = make_config(args)
  env = gym.make(args.env)
  env = CharToImage(env, env_conf)
  env = PrevActionsWrapper(env)
  model = NetHackModel(score_conf, device)
  model.load_state_dict(torch.load(model_path))
  model.eval()
  avg_rewards = []
  for eval_episode in range(10): # Average of 10 episodes
    done = False
    state, info = env.reset()
    rewards = []
    h, c = model.init_lstm()
    while not done:
      rgb_image = torch.from_numpy(state["rgb_image"]).unsqueeze(0).permute(0, 3, 1, 2).to(device) # B, C, H, W
      tl, bl = torch.from_numpy(state["tty_chars"][0, :]).long().unsqueeze(0).to(device), torch.from_numpy(state["tty_chars"][-2:, :]).float().unsqueeze(0).to(device)
      prev_action = torch.from_numpy(state["prev_actions"]).unsqueeze(dim=0).to(device)
      action_dist, value, _ = model(rgb_image, tl, bl, prev_action, h, c)
      action_dist = action_dist.squeeze()
      action = torch.multinomial(action_dist, num_samples=1)
      next_state, reward, terminated, truncated, info = env.step(action.cpu().item())
      done = terminated or truncated
      env.render()
      rewards.append(reward)
      if done:
        break
    episode_reward = sum(rewards)
    print(f"Reward on Eval episode: {eval_episode} was: {episode_reward}")
    avg_rewards.append(episode_reward)
  print(f"Average reward over 10 episodes was: {sum(avg_rewards) / len(avg_rewards)}")
  with open(env_conf["eval_path"], "w") as f:
    f.write(f"Average reward over 10 episodes was: {sum(avg_rewards) / len(avg_rewards)}")


def train_bc(model, optimizer, score_conf, env_conf):
  data_gen = dataset(env_conf["batch_size"], env_conf["seq_len"])
  cache_array = cache_ascii_char(env_conf)
  loss_fn = nn.CrossEntropyLoss()

  for minibatch in data_gen:
    h, c = model.init_lstm(env_conf["batch_size"])
    obs = torch.from_numpy(preprocess_dataset(minibatch, cache_array)["rgb_image"]).to(device)
    tl, bl = torch.from_numpy(minibatch["tty_chars"][:, :, 0, :]).long().to(device), torch.from_numpy(minibatch["tty_chars"][:, :, -2:, :]).float().to(device)
    prev_actions = minibatch["prev_actions"].to(device)
    encodings = model.encode(obs, tl, bl, prev_actions)
    action_dists, new_values, _ = model.recurrent(encodings, h, c)
    action_dists = action_dists.view(action_dists.shape[0]*action_dists.shape[1], -1)
    action_targets = minibatch["actions"].view(minibatch["actions"].shape[0]*minibatch["actions"].shape[1])
    loss = loss_fn(action_dists, action_targets)
    print(f"Loss was: {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train(score_conf, env_conf):

  if os.getenv("LOG"):
    run = wandb.init(project=env_conf["project_name"], config=env_conf)

  model = NetHackModel(score_conf, device)
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=env_conf["lr"])

  if env_conf["alg_type"] == "behavioural_cloning":
    train_bc(model, optimizer, score_conf, env_conf)
    exit()
  
  model.share_memory()
  mp.set_start_method('spawn')
  shared_queue = mp.Queue()
  
  model.to(device)

  workers = []
  for worker_id in range(env_conf["num_workers"]):
    worker = mp.Process(target=data_worker, args=(worker_id, env_conf, model, device, shared_queue, args))
    worker.start()
    workers.append(worker)

  print(f"Training on device {device}")
  print(f"Number of trainable parameters: {count_parameters(model)}")
  
  for training_step in range(env_conf["training_steps"]):
    st = time.perf_counter()

    all_rollouts = []
    for _ in range(env_conf["num_workers"]):
      rollout = shared_queue.get()
      if rollout: 
        all_rollouts.append(rollout)
        log_rewards = torch.sum(rollout["rewards"], dim=[0, 1]).item()
        if os.getenv("LOG"):
          wandb.log({"Reward": log_rewards})

    if not all_rollouts: 
      continue
    
    buffer = {
      key: torch.cat([r[key] for r in all_rollouts], dim=0)
      for key in all_rollouts[0]
    }

    returns = compute_returns(buffer["rewards"].cpu()).to(device)

    advantages = returns - buffer["values"]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    loss = ppo_update(model, optimizer, buffer["states_rgb"], buffer["states_tl"], buffer["states_bl"],
               buffer["actions"], buffer["prev_actions"], returns, advantages, buffer["log_probs_old"],
               buffer["hiddens"], buffer["cells"])
    
    if os.getenv("LOG"):
      wandb.log({"Loss": loss})

    et = time.perf_counter()
    with open(env_conf["log_path"], "a") as f:
      f.write(f"Reward: {log_rewards}\n")
      f.write(f"Loss: {loss.item()}\n")

    print(f"Single training step took: {et - st} seconds")
    print(f"Loss on episode: {training_step} was {loss.item():.4f}")

  for worker in workers:
    worker.terminate()
    
  torch.save(model.state_dict(), env_conf["default_model_path"])
  evaluate_model(env_conf["default_model_path"])

if __name__ == "__main__":
  score_conf, env_conf = make_configs()
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  if env_conf["mode"] == "eval":
    if not env_conf["checkpoint_path"]:
      raise Exception("No model path specified for evaluation")
    model_path = env_conf["checkpoint_path"]
    evaluate_model(model_path)
    exit()
    
  train(score_conf, env_conf)
  
