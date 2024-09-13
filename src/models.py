import torch
import torch.nn as nn
import torch.nn.functional as F

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
    o, (h, c) = self.core(encodings, (h.contiguous().to(self.device), c.contiguous().to(self.device)))
    action_dist = self.actor(o)
    value = self.critic(o)
    action_prob = F.softmax(action_dist, dim=-1)
    return action_prob, value, (h, c)

  def forward(self, image, tl, bl, prev_action, h, c):
    encodings = self.encode(image, tl, bl, prev_action)
    action_prob, value, (h, c) = self.recurrent(encodings, h, c)
    return action_prob, value, (h, c)
