import time
import os

def make_configs():
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
      "lstm_input": 889,
      "lstm_layers": 2,
      "lstm_hidden": 2048,
      "actor_out": 121
  }

  env_conf = {
    "project_name": "BC_training",   # Used for wandb
    "env_name": "NetHackChallenge-v0",
    "mode": "training",             # "training" or "eval"
    "checkpoint_path": "../other_param/pt_model_ckpts/lstm_bc.tar",        # if "eval" provide path to a model
    "default_model_path": f"../checkpoints/run-{time.strftime('%Y%m%d-%H%M%S')}.pt" if os.getenv("DEV") == "1" else  f"/workspace/NetHackRL/runlogs/run-{time.strftime('%Y%m%d-%H%M%S')}.pt",
    "model_storage": "../model_storage/model.safetensors" if os.getenv("DEV") == "1" else "/workspace/NetHackRL/model_storage/model.safetensors",
    "font_path": "../Hack-Regular.ttf" if os.getenv("DEV") == "1" else "/workspace/NetHackRL/Hack-Regular.ttf",
    "log_path": "../runlogs/log.txt" if os.getenv("DEV") == "1" else "/workspace/NetHackRL/runlogs/log.txt",
    "eval_path": "../runlogs/eval.txt" if os.getenv("DEV") == "1" else "/workspace/NetHackRL/runlogs/eval.txt",
    "alg_type": "behavioural_cloning",
    "max_norm": 0.5,
    "character": "mon-hum-neu-mal",
    "max_env_steps": 10000,
    "model_update_frequency": 20,
    "seq_len": 32,
    "batch_size": 128,
    "lr": 1e-4,
    "num_workers": 2,
    "training_steps": 3000,
    "obs_image_shape": (108, 108),
    "obs_tl_shape": 80,
    "obs_bl_shape": 80,
  }
  return score_conf, env_conf
