import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tinygrad import Tensor, nn
from models import NetHackModel
from configs import make_configs

def visualize_grid_of_filters(filters, title):
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
      ax.axis('off')  # Hide unused subplots
  
  plt.suptitle(title)
  fig.colorbar(im, ax=axes.ravel().tolist())
  plt.show()

def handle_4d_weights(weight_matrix):
  averaged_weights = np.mean(weight_matrix, axis=1)  # Reduce to 3D
  return averaged_weights


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
      visualize_grid_of_filters(activation, title=f"Activations: {layer_name}")

  def analyze_weights(self):
    for key, value in self.state_dict.items():
      weight_matrix = value.numpy()
      print(f"Visualizing {key} with shape {weight_matrix.shape}")

      if weight_matrix.ndim == 4:
        # 4D convolutions
        weight_matrix_3d = handle_4d_weights(weight_matrix)
        visualize_grid_of_filters(weight_matrix_3d, title=f"{key} (averaged)")

      elif weight_matrix.ndim == 2:
        # Linear layers 
        visualize_grid_of_filters(weight_matrix[np.newaxis, ...], title=key)

      elif weight_matrix.ndim == 1:
        print(f"Skipping visualization for {key} (1D bias)")
  
  def generate_html_report(self):
    html_content = "<html><head><title>NetHack Model Report</title></head><body>\n"
    html_content += "<h1>Weight Visualizations</h1>\n"

    for key, value in self.state_dict.items():
        weight_matrix = value.numpy()

        plt.figure(figsize=(8, 8))
        visualize_grid_of_filters(weight_matrix, title=key)
        image_path = f"visualizations/{key}.png"
        plt.savefig(image_path)
        plt.close()

        html_content += f"<h2>{key}</h2>\n"
        html_content += f'<img src="{image_path}" alt="{key}">\n'

    with open("nethack_report.html", "w") as file:
      file.write(html_content)

if __name__ == "__main__":
  visualizer = NetHackVisualizer("../checkpoints/run-20240927-224835.pt")
  visualizer.analyze_weights()
  visualizer.generate_html_report()

