from tinygrad import Tensor, nn, dtypes, TinyJit


############# CDGPT-5 ###############
class NetHackEncoder:
  def __init__(self, conv_channels, fc_dims, bl_conv_dims):
    self.screen_convs = []
    in_channels = 3
    kernels_strides = [(8, 6), (4, 2), (3, 2), (3, 1)]
    for layer, (kernel_size, stride) in zip(range(len(conv_channels)), kernels_strides):
      out_channels = conv_channels[layer]
      self.screen_convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
      self.screen_convs.append(Tensor.elu)
      in_channels = out_channels
    self.screen_out_fc = [nn.Linear(fc_dims["screen_in"], fc_dims["screen_out"]), Tensor.elu]

    self.tl_enc = [nn.Linear(fc_dims["tl_in"], fc_dims["tl_hidden"]), Tensor.elu, \
                   nn.Linear(fc_dims["tl_hidden"], fc_dims["tl_out"]), Tensor.elu]

    self.bl_convs = []
    for in_ch, out_ch, kernel, stride in bl_conv_dims:
      self.bl_convs.append(nn.Conv1d(in_ch, out_ch, kernel, stride=stride))
      self.bl_convs.append(Tensor.elu)

    self.bl_out_fc = [nn.Linear(fc_dims["bl_in"], fc_dims["bl_hidden"]), Tensor.elu, \
                      nn.Linear(fc_dims["bl_hidden"], fc_dims["bl_out"]), Tensor.elu]

  def __call__(self, x_rgb, tl, bl, prev_action):
    B = x_rgb.shape[0]
    x_rgb = x_rgb / 255.0
    conv_out = x_rgb.sequential(self.screen_convs)
    rgb_flatten = conv_out.reshape(B, -1)
    screen_enc = rgb_flatten.sequential(self.screen_out_fc)


    tl = tl.cast(dtypes.int64).one_hot(256).reshape(B, -1).cast(dtypes.float32)
    tl_enc = tl.sequential(self.tl_enc)

    bl = bl.view(B, -1)
    chars_normalised = (bl - 32.) / 96.
    numbers_mask = (bl > 44.) * (bl < 58.)
    digits_normalised = numbers_mask * (bl - 47.) / 10.
    bl_norm = Tensor.stack(chars_normalised, digits_normalised, dim=1)


    bl_enc = bl_norm.sequential(self.bl_convs).reshape(B, -1)
    bl_enc = bl_enc.sequential(self.bl_out_fc)

    pre_action_one_hot = prev_action.one_hot(121).view(B, 121)
    encodings = Tensor.cat(tl_enc, bl_enc, screen_enc, pre_action_one_hot, dim=1)
    return encodings


class LSTMCell:
  def __init__(self, input_size, hidden_size):
    self.weights_ih = Tensor.uniform(hidden_size * 4, input_size)
    self.weights_hh = Tensor.uniform(hidden_size * 4, hidden_size)
    self.bias_ih = Tensor.uniform(hidden_size * 4)
    self.bias_hh = Tensor.uniform(hidden_size * 4)

  def __call__(self, x, hc):
    gates = x.linear(self.weights_ih.T, self.bias_ih) + hc[:x.shape[0]].linear(self.weights_hh.T, self.bias_hh)

    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

    c = (f * hc[x.shape[0]:]) + (i * g)
    h = o * c.tanh()

    return Tensor.cat(h, c).realize()

class LSTM:
  def __init__(self, input_size, hidden_size, layers):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.layers = layers

    self.cells = [LSTMCell(input_size, hidden_size) if i == 0 else LSTMCell(hidden_size, hidden_size) for i in range(layers)]

  def __call__(self, x, hc):
    @TinyJit
    def _do_step(x_, hc_):
      return self.do_step(x_, hc_)

    if hc is None:
      hc = Tensor.zeros(self.layers, 2 * x.shape[1], self.hidden_size, requires_grad=False)
    
    output = None
    for t in range(x.shape[0]):
      hc = _do_step(x[t] + 1 - 1, hc) 
      if output is None:
        output = hc[-1:, :x.shape[1]]
      else:
        output = output.cat(hc[-1:, :x.shape[1]], dim=0).realize()

    return output, hc

  def do_step(self, x, hc):
    new_hc = [x]
    for i, cell in enumerate(self.cells):
      new_hc.append(cell(new_hc[i][:x.shape[0]], hc[i]))
    return Tensor.stack(*new_hc[1:]).realize()

class NetHackModel:
  def __init__(self, config):
    self.config = config
    self.encoder = NetHackEncoder(conv_channels=config["conv_channels"], fc_dims=config["fc_dims"], bl_conv_dims=config["bl_conv_dims"])
    self.core = LSTM(config["lstm_input"], config["lstm_hidden"], config["lstm_layers"])
    self.actor = nn.Linear(config["lstm_hidden"], config["actor_out"])

  def init_lstm(self, batch_size=1):
    return Tensor.zeros((batch_size, self.config["lstm_hidden"])), Tensor.zeros((batch_size, self.config["lstm_hidden"]))

  def encode(self, image, tl, bl, prev_action):
    assert len(image.shape) == 5
    image_enc = image.view(image.shape[0] * image.shape[1], image.shape[4], image.shape[2], image.shape[3])

    encodings = self.encoder(image_enc, tl, bl, prev_action)
    encodings = encodings.view(image.shape[0], image.shape[1], -1)

    return encodings

  def recurrent(self, encodings, hc):
    encodings = encodings.transpose(0, 1)
    o, hc = self.core(encodings, hc)
    o = o.transpose(0, 1)

    action_logits = self.actor(o)
    return action_logits, hc
    
  def __call__(self, image, tl, bl, prev_actions, state=None):
    logits, new_state = self.recurrent(self.encode(image, tl, bl, prev_actions), state)
    return logits, new_state


