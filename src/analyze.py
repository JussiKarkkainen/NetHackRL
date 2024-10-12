import numpy as np
import nle
import gymnasium as gym
from preprocessing import CharToImage, PrevActionsWrapper, cache_ascii_char, preprocess_dataset
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tinygrad import Tensor, nn
from models import NetHackModel
from configs import make_configs

def visualize_grid_of_filters(filters, title, save_path):
  num_filters = filters.shape[0]
  filter_size = filters.shape[1:]
  grid_rows = int(np.sqrt(num_filters))
  grid_cols = int(np.ceil(num_filters / grid_rows))
  fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 12))

  if num_filters == 1:
    axes = np.array([axes])

  cmap = plt.cm.RdBu_r
  vmin = np.nanmin(filters)
  vmax = np.nanmax(filters)

  for i, ax in enumerate(axes.flat):
    if i < num_filters:
      filter_data = filters[i]
      nan_mask = np.isnan(filter_data)
      filter_data = np.ma.array(filter_data, mask=nan_mask)

      im = ax.imshow(filter_data, cmap=cmap, vmin=vmin, vmax=vmax)
      cmap.set_bad(color='black')
      ax.axis('off')
    else:
      ax.axis('off')
  
  plt.suptitle(title)
  fig.colorbar(im, ax=axes.ravel().tolist())

  if save_path:
    plt.savefig(save_path)
  plt.close(fig)

def handle_4d_weights(weight_matrix):
  averaged_weights = np.mean(weight_matrix, axis=1)  # Reduce to 3D
  return averaged_weights

def compute_weight_statistics(weight_matrix):
  stats = {}

  num_nans = np.isnan(weight_matrix).sum()

  non_nan_weights = weight_matrix[~np.isnan(weight_matrix)]
  max_val = np.max(non_nan_weights) if non_nan_weights.size > 0 else float('nan')
  min_val = np.min(non_nan_weights) if non_nan_weights.size > 0 else float('nan')
  mean_val = np.mean(non_nan_weights) if non_nan_weights.size > 0 else float('nan')
  std_val = np.std(non_nan_weights) if non_nan_weights.size > 0 else float('nan')

  total_elements = weight_matrix.size
  nan_percentage = (num_nans / total_elements) * 100 if total_elements > 0 else 0

  stats['num_nans'] = num_nans
  stats['nan_percentage'] = nan_percentage
  stats['max'] = max_val
  stats['min'] = min_val
  stats['mean'] = mean_val
  stats['std'] = std_val

  return stats

def add_weight_statistics_to_html(stats, layer_name):
  stats_html = f"""
  <h3>Statistics for {layer_name}</h3>
  <ul>
      <li>Number of NaNs: {stats['num_nans']}</li>
      <li>Percentage of NaNs: {stats['nan_percentage']:.2f}%</li>
      <li>Maximum value: {stats['max']}</li>
      <li>Minimum value: {stats['min']}</li>
      <li>Mean value: {stats['mean']}</li>
      <li>Standard deviation: {stats['std']}</li>
  </ul>
  """
  return stats_html


def capture_activations(layer, activations_dict, layer_name):
  def wrapper(*args, **kwargs):
    output = layer(*args, **kwargs)
    activations_dict[layer_name] = output.numpy()
    return output
  return wrapper

class NetHackVisualizer:
  def __init__(self, model_path):
    score_conf, env_conf = make_configs()
    self.model = NetHackModel(score_conf, use_critic=False)
    self.state_dict = nn.state.safe_load(model_path)
    self.activations = {}

  def activations_model(self):
    self.model.encoder.__call__ = capture_activations(self.model.encoder.__call__)
    self.model.core.__call__ = capture_activations(self.model.core.__call__)

  def analyze_activations(self, inputs):
    self.activations_model()
    h, c = self.model.init_lstm(batch_size=inputs.shape[0])
    encodings = self.model.encode(*inputs)
    log_prob, value, (h, c) = self.model.recurrent(encodings, h, c)
    for layer_name, activation in self.activations.items():
      print(f"Visualizing activations for {layer_name} with shape {activation.shape}")
      visualize_grid_of_filters(activation, title=f"Activations: {layer_name}", save_path=f"../reports/{layer_name_act}.png")

  def analyze_weights(self):
    for key, value in self.state_dict.items():
      weight_matrix = value.numpy()
      print(f"Visualizing {key} with shape {weight_matrix.shape}")

      if weight_matrix.ndim == 4:
        # 4D convolutions
        weight_matrix_3d = handle_4d_weights(weight_matrix)
        visualize_grid_of_filters(weight_matrix_3d, title=f"{key} (averaged)", save_path=f"../reports/{key}.png")

      elif weight_matrix.ndim == 2:
        # Linear layers 
        visualize_grid_of_filters(weight_matrix[np.newaxis, ...], title=key, save_path=f"../reports/{key}.png")

      elif weight_matrix.ndim == 1:
        print(f"Skipping visualization for {key} (1D bias)")
  
  def generate_html_report(self):
    html_content = "<html><head><title>NetHack Model Report</title></head><body>\n"
    html_content += "<h1>Weight Visualizations</h1>\n"

    for key, value in self.state_dict.items():
        weight_matrix = value.numpy()
        stats = compute_weight_statistics(weight_matrix)

        html_content += add_weight_statistics_to_html(stats, key)

        image_path = f"{key}.png"

        html_content += f"<h2>{key}</h2>\n"
        html_content += f'<img src="{image_path}" alt="{key}">\n'

    html_content += "</body></html>"

    with open("../reports/nethack_report.html", "w") as file:
      file.write(html_content)

from flask import Flask, render_template, jsonify

app = Flask(__name__)

class _NetHackGame:
  def __init__(self, model_path):
    score_conf, env_conf = make_configs()
    self.env = gym.make(env_conf["env_name"], character=env_conf["character"])
    self.env = CharToImage(self.env, env_conf)
    self.env = PrevActionsWrapper(self.env)

    self.model = NetHackModel(score_conf, use_critic=False)
    nn.state.load_state_dict(self.model, nn.state.safe_load(model_path))

    self.state, info = self.env.reset()
    self.h, self.c = self.model.init_lstm()
    self.done = False

  def step_game(self):
    obs = Tensor(self.state["rgb_image"]).unsqueeze(dim=0).transpose(1, 3)
    tl = Tensor(self.state["tty_chars"][0, :]).unsqueeze(dim=0)
    bl = Tensor(self.state["tty_chars"][-2:, :]).unsqueeze(dim=0).float()
    prev_actions = Tensor(self.state["prev_actions"])

    log_probs, _, (h_list, c_list) = self.model(obs, tl, bl, prev_actions, h, c)
    log_probs_s = log_probs.squeeze()
    u = Tensor.uniform(shape=log_probs_s.shape)
    action = Tensor.argmax(log_probs_s - Tensor.log(-Tensor.log(u)), axis=-1)
    
    next_state, reward, terminated, truncated, info = self.env.step(action.item())
    self.done = terminated or truncated
    self.state = next_state
    return state, action.item()

@app.route('/init')
def init():
  global NetHackGame
  NetHackGame = _NetHackGame("../checkpoints/run-20241009-223400.pt")
  init_state = NetHackGame.state
  print(init_state["rgb_image"].shape)
  return jsonify({"state": init_state["rgb_image"].tolist()})

@app.route('/')
def main():
  return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)

'''
if __name__ == "__main__":
  visualizer = NetHackVisualizer("../checkpoints/run-20241009-223400.pt")
  visualizer.analyze_weights()
  visualizer.generate_html_report()
'''
