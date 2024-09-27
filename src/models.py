from tinygrad import Tensor, nn

class NetHackEncoder:
  def __init__(self, conv_channels, fc_dims, bl_conv_dims):
    self.screen_convs = []
    in_channels = 3
    kernels_strides = [(8, 6), (4, 2), (3, 2), (3, 1)]
    for layer, (kernel_size, stride) in zip(range(len(conv_channels)), kernels_strides):
      out_channels = conv_channels[layer]
      self.screen_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
      if layer != len(conv_channels) - 1:
        self.screen_convs.append(Tensor.elu)
      in_channels = out_channels
    self.screen_out_fc = nn.Linear(fc_dims["screen_in"], fc_dims["screen_out"])

    self.tl_enc = [nn.Linear(fc_dims["tl_in"], fc_dims["tl_hidden"]), Tensor.elu, \
                   nn.Linear(fc_dims["tl_hidden"], fc_dims["tl_out"])]

    self.bl_convs = []
    for in_ch, out_ch, kernel, stride in bl_conv_dims:
      self.bl_convs.append(nn.Conv1d(in_ch, out_ch, kernel, stride=stride))
      self.bl_convs.append(Tensor.elu)

    self.bl_out_fc = [nn.Linear(fc_dims["bl_in"], fc_dims["bl_hidden"]), Tensor.elu, \
                      nn.Linear(fc_dims["bl_hidden"], fc_dims["bl_out"])]

    
  def __call__(self, x_rgb, tl, bl, prev_action):
    B = x_rgb.shape[0]
    x_rgb = x_rgb / 255.0
    conv_out = x_rgb.sequential(self.screen_convs)
    rgb_flatten = conv_out.reshape(B, -1)
    screen_enc = self.screen_out_fc(rgb_flatten)

    tl = tl.one_hot(256).reshape(B, -1)
    tl_enc = tl.sequential(self.tl_enc)

    # https://github.com/BartekCupial/sample-factory/blob/master/sf_examples/nethack/models/chaotic_dwarf.py
    bl = bl.view(B, -1)
    chars_normalised = (bl - 32) / 96
    numbers_mask = (bl > 44) * (bl < 58)
    digits_normalised = numbers_mask * (bl - 47) / 10
    bl_norm = Tensor.stack(chars_normalised, digits_normalised, dim=1)

    bl_enc = bl_norm.sequential(self.bl_convs).reshape(B, -1)
    bl_enc = bl_enc.sequential(self.bl_out_fc)

    pre_action_one_hot = prev_action.one_hot(23).view(B, 23)
    encodings = Tensor.cat(tl_enc, screen_enc, bl_enc, pre_action_one_hot, dim=-1)
    return encodings

class NetHackModel:
  def __init__(self, config, use_critic=True):
    self.config = config
    self.encoder = NetHackEncoder(conv_channels=config["conv_channels"], fc_dims=config["fc_dims"], bl_conv_dims=config["bl_conv_dims"])
    self.core = nn.LSTMCell(input_size=config["lstm_input"], hidden_size=config["lstm_hidden"])
    self.actor = nn.Linear(config["lstm_hidden"], config["actor_out"])
    self.critic = nn.Linear(config["lstm_hidden"], 1) if use_critic else None

  def init_lstm(self, batch_size=1):
    return Tensor.zeros((batch_size, self.config["lstm_hidden"])), Tensor.zeros((batch_size, self.config["lstm_hidden"]))

  def __call__(self, image, tl, bl, prev_action, h, c):
    is_batched = len(image.shape) == 5
    image_enc = image.view(image.shape[0] * image.shape[1], image.shape[4], image.shape[2], image.shape[3]) if is_batched else image

    encodings = self.encoder(image_enc, tl, bl, prev_action)
    encodings = encodings.view(image.shape[0], image.shape[1], -1) if is_batched else encodings.unsqueeze(0)
    
    h, c = h.contiguous(), c.contiguous()
    o = []
    for t in range(encodings.shape[1]):   # B, T, ...
      h, c = self.core(encodings[:, t, ...], (h, c))
      o.append(h.unsqueeze(1))

    o = Tensor.cat(*o, dim=1)
    action_dist = self.actor(o)
    value = self.critic(o) if self.critic is not None else None
    action_prob = action_dist.log_softmax(axis=-1)
    return action_prob, value, (h, c)

