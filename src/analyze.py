import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tinygrad import Tensor, nn
from models import NetHackModel
from configs import make_configs

def visualize_grid_of_filters(filters, title):
    num_filters = filters.shape[0]
    filter_size = filters.shape[1:]

    # Determine grid size (e.g., 8x4 grid for 32 filters)
    grid_rows = int(np.sqrt(num_filters))
    grid_cols = int(np.ceil(num_filters / grid_rows))

    # Create a figure with subplots
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(12, 12))

    # Ensure axes is always 2D
    if num_filters == 1:
        axes = np.array([axes])

    # Normalize for color mapping
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
    # For 4D weights (e.g., (out_channels, in_channels, height, width)),
    # we average across input channels (dim 1) to reduce to 3D
    averaged_weights = np.mean(weight_matrix, axis=1)  # Resulting shape: (out_channels, height, width)
    return averaged_weights

def analyze_weights(model_path):
    score_conf, env_conf = make_configs()
    model = NetHackModel(score_conf, use_critic=False)
    state_dict = nn.state.safe_load(model_path)
    
    for key, value in state_dict.items():
        weight_matrix = value.numpy()
        print(f"Visualizing {key} with shape {weight_matrix.shape}")
        
        if weight_matrix.ndim == 4:
            # Handle 4D convolutional weights
            weight_matrix_3d = handle_4d_weights(weight_matrix)
            visualize_grid_of_filters(weight_matrix_3d, title=f"{key} (averaged)")

        elif weight_matrix.ndim == 2:
            # Direct visualization for 2D matrices (e.g., fully connected layers)
            visualize_grid_of_filters(weight_matrix[np.newaxis, ...], title=key)  # Add extra dim for consistency

        elif weight_matrix.ndim == 1:
            # Skip 1D biases
            print(f"Skipping visualization for {key} (1D bias)")

if __name__ == "__main__":
    analyze_weights("../checkpoints/run-20240927-224835.pt")

