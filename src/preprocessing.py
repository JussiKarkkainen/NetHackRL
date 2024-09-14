import nle
import gymnasium as gym
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import time

COLORS = ["#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080",
          "#808080", "#C0C0C0", "#FF0000", "#00FF00", "#FFFF00", "#0000FF", "#FF00FF",
          "#00FFFF", "#FFFFFF"]

def cache_ascii_char(env_conf, rescale_font_size=9):
  font = ImageFont.truetype(env_conf["font_path"], 9)
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

def preprocess_dataset(obs, cache_array):
  B, T = obs["tty_chars"].shape[:2]
  chars = obs["tty_chars"][:, :, 1:-2, :]
  colors = np.clip(obs["tty_colors"], 0, 15)
  
  # 11*chars.shape[0], 6*chars.shape[1] for full image
  # 11 for better looking image
  pixel_obs = np.zeros((B, T, 9*12, 9*12, 3), dtype=np.float32)

  for b in range(B):
    for t in range(T):
      for i in range(12): # chars.shape[0] for full screen
        for j in range(12): # chars.shape[1] for full screen
          color = colors[b, t, i, j]
          char = cache_array[chars[b, t, i, j]][color]
          pixel_obs[b, t, i*9:(i+1)*9, j*9:(j+1)*9, :] = char

  return pixel_obs

class CharToImage(gym.Wrapper):
  def __init__(self, env, env_conf):
    super().__init__(env)
    self.cache_array = cache_ascii_char(env_conf)

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
    obs, reward, terminated, truncated, info = self.env.step(action)
    self._render_to_image(obs)
    return obs, reward, terminated, truncated, info

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    self._render_to_image(obs)
    return obs, info

class PrevActionsWrapper(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.prev_action = 0
    obs_spaces = {"prev_actions": self.env.action_space}
    obs_spaces.update([(k, self.env.observation_space[k]) for k in self.env.observation_space])
    self.observation_space = gym.spaces.Dict(obs_spaces)

  def reset(self, **kwargs):
    self.prev_action = 0
    obs, info = self.env.reset(**kwargs)
    obs["prev_actions"] = np.array([self.prev_action])
    return obs, info

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    self.prev_action = action
    obs["prev_actions"] = np.array([self.prev_action])
    return obs, reward, terminated, truncated, info
