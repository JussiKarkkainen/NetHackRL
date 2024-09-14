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
      "lstm_input": 791,
      "lstm_hidden": 512,
      "actor_out": 23
  }

  env_conf = {
    "project_name": "BC_training",   # Used for wandb
    "env_name": "NetHackScore-v0",
    "mode": "training",             # "training" or "eval"
    "checkpoint_path": None,        # if "eval" provide path to a model
    "default_model_path": f"/workspace/runlogs/run-{time.strftime('%Y%m%d-%H%M%S')}.pt",
    "font_path": "../Hack-Regular.ttf" if os.getenv("DEV") == "1" else "/workspace/PPO_nethack/Hack-Regular.ttf",
    "log_path": "../runlogs/log.txt" if os.getenv("DEV") == "1" else "/workspace/runlogs/log.txt",
    "eval_path": "../runlogs/eval.txt" if os.getenv("DEV") == "1" else "/workspace/runlogs/eval.txt",
    "alg_type": "behavioural_cloning",
    "max_env_steps": 10000,
    "seq_len": 256,
    "lr": 3e-4,
    "num_workers": 2,
    "training_steps": 500,
    "obs_image_shape": (108, 108),
    "obs_tl_shape": 80,
    "obs_bl_shape": 80,
  }
  return score_conf, env_conf
