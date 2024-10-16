import nle
import gymnasium as gym
import numpy as np
from tinygrad import Tensor, nn, helpers, Device
import wandb
import time
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import reduce
from preprocessing import CharToImage, PrevActionsWrapper, cache_ascii_char, preprocess_dataset
from nld_aa_pretraining import dataset
from models import NetHackModel
from configs import make_configs
from data_workers import data_worker
from algs import ppo_update, compute_returns, bc_update, bc_accuracy

def count_parameters(model):
  return sum(p.numel() for p in nn.state.get_parameters(model) if p.requires_grad)

@Tensor.test()
def render_episode(model_path, score_conf, env_conf):
  env = gym.make(env_conf["env_name"], character=env_conf["character"])
  env = CharToImage(env, env_conf)
  env = PrevActionsWrapper(env)
  
  model = NetHackModel(score_conf, use_critic=False)
  nn.state.load_state_dict(model, nn.state.safe_load(model_path))

  state, info = env.reset()
  h, c = model.init_lstm()
  done = False
  step = 0
  while not done:
    obs = Tensor(state["rgb_image"]).unsqueeze(dim=0).transpose(1, 3)
    tl = Tensor(state["tty_chars"][0, :]).unsqueeze(dim=0)
    bl = Tensor(state["tty_chars"][-2:, :]).unsqueeze(dim=0).float()
    prev_actions = Tensor(state["prev_actions"])

    log_probs, _, (h_list, c_list) = model(obs, tl, bl, prev_actions, h, c)
    log_probs_s = log_probs.squeeze()
    u = Tensor.uniform(shape=log_probs_s.shape)
    action = Tensor.argmax(log_probs_s - Tensor.log(-Tensor.log(u)), axis=-1)
    print(action.item())
    
    #env.render()
    next_state, reward, terminated, truncated, info = env.step(action.item())
    done = terminated or truncated
    state = next_state
    step += 1

    if step % 100 == 0:
      env.render()
      plt.imshow(state["rgb_image"], interpolation='nearest')
      plt.show()
      raise Exception


@Tensor.test()
def evaluate_model(model_path, score_conf, env_conf):
  envs = [gym.make(env_conf["env_name"], character=env_conf["character"]) for _ in range(10)]
  envs = [CharToImage(env, env_conf) for env in envs]
  envs = [PrevActionsWrapper(env) for env in envs] 

  model = NetHackModel(score_conf, use_critic=False)
  nn.state.load_state_dict(model, nn.state.safe_load(model_path))

  avg_rewards = []
  states = []
  dones = [False] * 10
  h_list, c_list = zip(*[model.init_lstm() for _ in range(10)])

  for env in envs:
    state, _ = env.reset()
    states.append(state)
  
  rewards_list = [[] for _ in range(10)]

  step = 0
  while not all(dones):
    obs_tensors, tl_tensors, bl_tensors, prev_actions_tensors = [], [], [], []
    for i, (state, done) in enumerate(zip(states, dones)):
      if not done:
        obs = Tensor(state["rgb_image"]).unsqueeze(dim=0).transpose(1, 3)
        tl = Tensor(state["tty_chars"][0, :]).unsqueeze(dim=0)
        bl = Tensor(state["tty_chars"][-2:, :]).unsqueeze(dim=0).float()
        prev_actions = Tensor(state["prev_actions"])

        obs_tensors.append(obs)
        tl_tensors.append(tl)
        bl_tensors.append(bl)
        prev_actions_tensors.append(prev_actions)

    if obs_tensors:
      obs_batch = Tensor.stack(*obs_tensors).transpose(2, 4)
      tl_batch = Tensor.stack(*tl_tensors)
      bl_batch = Tensor.stack(*bl_tensors)
      prev_actions_batch = Tensor.stack(*prev_actions_tensors)
      h_batch = Tensor.stack(*h_list).squeeze()
      c_batch = Tensor.stack(*c_list).squeeze()

      log_probs, _, (h_list, c_list) = model(obs_batch, tl_batch, bl_batch, prev_actions_batch.reshape(10, 1), h_batch, c_batch)
      log_probs_s = log_probs.squeeze()
      u = Tensor.uniform(shape=log_probs_s.shape)
      actions = Tensor.argmax(log_probs_s - Tensor.log(-Tensor.log(u)), axis=-1)

      for i, env in enumerate(envs):
        if not dones[i]:
          next_state, reward, terminated, truncated, info = env.step(actions[i].item())
          rewards_list[i].append(reward)
          dones[i] = terminated or truncated
          if not dones[i]:
            states[i] = next_state  # Update state for the environment
          else:
            avg_rewards.append(sum(rewards_list[i]))
    step += 1
    if step % 10 == 0:
      print(f"Number of environments still running: {len([d for d in dones if d == False])}")


  print(f"Average reward over 10 episodes was: {sum(avg_rewards) / len(avg_rewards)}")
  with open(env_conf["eval_path"], "w") as f:
    f.write(f"Average reward over 10 episodes was: {sum(avg_rewards) / len(avg_rewards)}")
  

@Tensor.train()
def train_bc(model, score_conf, env_conf):
  data_gen = dataset(env_conf["batch_size"], env_conf["seq_len"])
  cache_array = cache_ascii_char(env_conf)
  
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=env_conf["lr"])
  
  print(f"Number of trainable parameters: {count_parameters(model)}")

  cnt = 0
  for minibatch in data_gen:
    if cnt == env_conf["training_steps"]:
      print("Trained for {env_conf['training_steps']}, ending training")
      break

    h, c = model.init_lstm(env_conf["batch_size"])
    
    obs = Tensor(preprocess_dataset(minibatch, cache_array)["rgb_image"])
    tl, bl = Tensor(minibatch["tty_chars"][:, :, 0, :]), Tensor(minibatch["tty_chars"][:, :, -2:, :]).float()
    action_targets = minibatch["actions"].view(minibatch["actions"].shape[0]*minibatch["actions"].shape[1], 1)
    prev_actions = minibatch["prev_actions"]
    
    with helpers.Timing("Time for update step: "):
      loss = bc_update(model, optimizer, h, c, obs, tl, bl, action_targets, prev_actions)

    if cnt % 100 == 0:
      accuracy = bc_accuracy(model, h, c, obs, tl, bl, action_targets, prev_actions)
      print(f"Accuracy on minibatch: {cnt} was {accuracy.item():.2f}%")
      if os.getenv("LOG"):
        wandb.log({"Accuracy": accuracy.item()})

    if os.getenv("LOG"):
      wandb.log({"Loss": loss.item()})


    print(f"Loss on minibatch: {cnt} was {loss.item():.4f}")
    cnt += 1
  
  nn.state.safe_save(nn.state.get_state_dict(model), env_conf["default_model_path"])
  evaluate_model(env_conf["default_model_path"], score_conf, env_conf)

def train_ppo(model, score_conf, env_conf):
  mp.set_start_method('spawn')
  data_queue = mp.Queue()

  if os.path.exists(env_conf["model_storage"]):
    os.remove(env_conf["model_storage"])
  os.makedirs(os.path.split(env_conf["model_storage"])[0], exist_ok=True)
  nn.state.safe_save(nn.state.get_state_dict(model), env_conf["model_storage"])
  
  optimizer = nn.optim.Adam(nn.state.get_parameters(model), lr=env_conf["lr"])

  workers = []
  for _ in range(env_conf["num_workers"]):
    worker = mp.Process(target=data_worker, args=(env_conf, score_conf, data_queue))
    worker.start()
    workers.append(worker)

  print(f"Number of trainable parameters: {count_parameters(model)}")
  
  for training_step in range(env_conf["training_steps"]):
    st = time.perf_counter()

    all_rollouts = []
    for _ in range(env_conf["num_workers"]):
      rollout = data_queue.get()
      if rollout: 
        all_rollouts.append(rollout)
        log_rewards = Tensor.sum(rollout["rewards"], axis=[0, 1]).item()
        if os.getenv("LOG"):
          wandb.log({"Reward": log_rewards})

    if not all_rollouts: 
      continue
    
    buffer = {
      key: Tensor.cat(*[r[key] for r in all_rollouts], dim=0)
      for key in all_rollouts[0]
    }

    returns = compute_returns(buffer["rewards"])

    advantages = returns.squeeze() - buffer["values"].squeeze()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    loss = ppo_update(model, optimizer, buffer["states_rgb"], buffer["states_tl"], buffer["states_bl"],
               buffer["actions"], buffer["prev_actions"], returns, advantages, buffer["log_probs_old"],
               buffer["hiddens"], buffer["cells"])
    
    if training_step % env_conf["model_update_frequency"] == 0:
      nn.state.safe_save(nn.state.get_state_dict(model), env_conf["model_storage"])
    
    if os.getenv("LOG"):
      wandb.log({"Loss": loss.item()})

    et = time.perf_counter()
    with open(env_conf["log_path"], "a") as f:
      f.write(f"Reward: {log_rewards}\n")
      f.write(f"Loss: {loss.item()}\n")

    print(f"Single training step took: {et - st} seconds")
    print(f"Loss on episode: {training_step} was {loss.item():.4f}")

  for worker in workers:
    worker.terminate()
    
  nn.state.safe_save(nn.state.get_state_dict(model), env_conf["default_model_path"])
  evaluate_model(env_conf["default_model_path"], score_conf, env_conf)

if __name__ == "__main__":
  score_conf, env_conf = make_configs()
  
  if env_conf["mode"] == "eval":
    if not env_conf["checkpoint_path"]:
      raise Exception("No model path specified for evaluation")
    model_path = env_conf["checkpoint_path"]
    if os.getenv("RENDER") == "1":
      render_episode(model_path, score_conf, env_conf)
    else:
      evaluate_model(model_path, score_conf, env_conf)
    exit()
    
  if os.getenv("LOG"):
    run = wandb.init(project=env_conf["project_name"], config=env_conf)

  print(f"Training on device: {Device.DEFAULT}")

  match env_conf["alg_type"]:
    case "behavioural_cloning":
      model = NetHackModel(score_conf, use_critic=False)
      train_bc(model, score_conf, env_conf)
    case "ppo":
      model = NetHackModel(score_conf, use_critic=True)
      train_ppo(model, score_conf, env_conf)
    case _:
      raise Exception("Invalid 'alg_type' provided in the config")
  
