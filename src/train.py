import nle
import gymnasium as gym
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
from preprocessing import CharToImage, PrevActionsWrapper
from models import NetHackModel
from configs import score_conf, make_config
from data_workers import data_worker
from algs import ppo_update, compute_returns

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(args):
  env_conf = make_config(args)

  if os.getenv("LOG"):
    run = wandb.init(project="PPO-Baseline", config = env_conf)

  model = NetHackModel(score_conf, device)
  model.share_memory()
  model.train()
  optimizer = torch.optim.Adam(model.parameters(), lr=env_conf["lr"])
  
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
      action_dist, valuem, _ = model(rgb_image, tl, bl, prev_action, h, c)
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

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--env", help="Enter the environment, default: NetHackScore-v0", 
                      default="NetHackScore-v0")
  parser.add_argument("--eval", help="Eval a given model, include a path to model file", 
                      action="store_true")
  parser.add_argument("--checkpoint_path", help="Path to a model", default=None)
  parser.add_argument("--training_steps", type=int, help="Number of training steps, default: 500", default=500)
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
  
