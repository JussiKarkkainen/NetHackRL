import nle
import gymnasium as gym
import numpy as np
from tinygrad import Tensor
import cv2
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from preprocess_cython.preprocess import preprocess_char_image_cython
import time

from nle.nethack import tty_render

COLORS = ["#000000", "#800000", "#008000", "#808000", "#000080", "#800080", "#008080",
          "#808080", "#C0C0C0", "#FF0000", "#00FF00", "#FFFF00", "#0000FF", "#FF00FF",
          "#00FFFF", "#FFFFFF"]

def cache_ascii_char(env_conf, rescale_font_size=(9,9)):
  font = ImageFont.truetype(env_conf["font_path"], 9)
  dummy_text = "".join([(chr(i) if chr(i).isprintable() else " ") for i in range(256)])
  bboxes = np.array([font.getbbox(char) for char in dummy_text])
  image_width = bboxes[:, 2].max() # 6
  image_height = bboxes[:, 3].max() # 11

  char_width, char_height = rescale_font_size

  char_array = np.zeros((256, 16, char_height, char_width, 3), dtype=np.uint8)

  for color_index in range(16):
    for char_index in range(256):
      char = dummy_text[char_index]
      
      image = Image.new("RGB", (image_width, image_height))
      image_draw = ImageDraw.Draw(image)
      image_draw.rectangle((0, 0, image_width, image_height), fill=(0, 0, 0))
      
      _, _, width, height = font.getbbox(char)
      
      image_draw.text((image_width - width, image_height - height), char, font=font, fill=COLORS[color_index])
      
      arr = np.array(image).copy()
      if rescale_font_size:
        arr = cv2.resize(arr, rescale_font_size, interpolation=cv2.INTER_AREA)
      
      char_array[char_index, color_index] = arr
  return char_array
    
def preprocess_dataset(obs, cache_array):
  B, T = obs["tty_chars"].shape[:2]
  out_height_char = 12
  out_width_char = 12

  # Initialize the output array
  pixel_obs = np.zeros((B, T, 9 * 12, 9 * 12, 3), dtype=np.float32)

  for b in range(B):
    for t in range(T):
      out_image = np.zeros((out_height_char*9, out_width_char*9, 3), dtype=np.uint8)
      chars = obs["tty_chars"][b, t, 1:-2, :]  
      colors = np.clip(obs["tty_colors"][b, t, 1:-2, :], 0, 15)

      center_y = obs["tty_cursor"][b, t, 0:1]
      center_x = obs["tty_cursor"][b, t, 1:2]
      offset_h = center_y.astype(np.int32) - 6
      offset_w = center_x.astype(np.int32) - 6

      preprocess_char_image_cython(out_image, chars, colors, out_width_char, out_height_char, 
                      offset_h, offset_w, cache_array)
        
      pixel_obs[b, t] = out_image

  obs["rgb_image"] = pixel_obs

  # Handling previous actions
  prev_actions = Tensor.zeros_like(obs["actions"]).contiguous()
  prev_actions[:, 1:] = obs["actions"][:, :-1]
  obs["prev_actions"] = prev_actions

  return obs

def preprocess_test(out_image, chars, colors, out_width_char, out_height_char,
                    offset_h, offset_w, cache_array):
  char_height = cache_array.shape[3]
  char_width = cache_array.shape[4]
  for h in range(out_height_char):
    h_char = h + offset_h
    if h_char < 0 or h_char >= chars.shape[0]:
      continue
    for w in range(out_width_char):
      w_char = w + offset_w
      if w_char < 0 or w_char >= chars.shape[1]:
        continue
      char = chars[h_char, w_char]
      color = colors[h_char, w_char]
      h_pixel = h * char_height
      w_pixel = w * char_width
      out_image[h_pixel:h_pixel + 9, w_pixel:w_pixel + 9, :] = cache_array[char, color]

  return out_image

class CharToImage(gym.Wrapper):
  def __init__(self, env, env_conf):
    super().__init__(env)
    self.cache_array = cache_ascii_char(env_conf)
    self.config = env_conf

  def _render_to_image(self, obs):
    chars = obs["tty_chars"][1:-2, :]
    colors = np.clip(obs["tty_colors"][1:-2, :], 0, 15)
    center_y, center_x = obs["tty_cursor"]
    offset_h = center_y.astype(np.int32) - 6
    offset_w = center_x.astype(np.int32) - 6
    out_height_char = 12
    out_width_char = 12
    out_image = np.zeros((out_height_char*9, out_width_char*9, 3), dtype=np.uint8)
    obs["rgb_image"] = preprocess_char_image_cython(out_image, chars, colors, out_width_char, 
                                       out_height_char, offset_h, offset_w, self.cache_array)

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
