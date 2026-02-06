import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import warnings
import h5py
import ipywidgets as widgets
import matplotlib as mpl

from scipy.interpolate import interpn
from matplotlib.animation import FuncAnimation
from scipy.io.matlab.miobase import MatReadWarning
from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA
from omegaconf import DictConfig

import torch.nn as nn

# Re-export DynamicData from data module for backward compatibility
from .data import DynamicData, DataConfig, resolve_data_path

def get_pca_error(data, k, ord=2):
  pca = PCA(n_components=k)

  principalComponents = pca.fit_transform(data)
  rdata = pca.inverse_transform(principalComponents)
  rerror = np.mean(np.linalg.norm(rdata - data, axis=1, ord=ord) / np.linalg.norm(data, axis=1, ord=ord))

  return rerror

def num_params(model):
  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  return params

def barycentric_contraction(points, scale):
  center = np.mean(points, axis=0)
  return scale * (points - center)

def linear_interpolate(arr, orig, new): # look to make this parallel
  return np.apply_along_axis(lambda row: interpn([orig], row, new), 1, arr)

def cubic_interpolate(arr, orig, new): # look to make this parallel
  return np.apply_along_axis(lambda row: interpn([orig], row, new, method="cubic"), 1, arr)

def pde_system_video(array, t, T=1, noise=0):
  fig, ax = plt.subplots()
  noisedarray = array.copy()
  noisedarray[:, 1:, :] += np.random.normal(scale=noise, size=noisedarray[:, 1:, :].shape)

  def update(frame):
    ax.clear()
    ax.plot(array[frame, 0, :], 'r', label="t=0")
    ax.plot(noisedarray[frame, t, :], 'b', label=f"t={t / (array.shape[1] - 1) * T}")
    ax.set_title(frame)
    ax.legend()
    
    if frame % 10 == 0:
      print(frame)

  num_frames = 50
  ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)
  ani.save("output_video.mp4", writer="ffmpeg", fps=5)

def link_3d_rotations(axes):
    """
    Syncs the 3D camera rotation (elev/azim) between multiple 3D axes
    whenever one of them is clicked.
    """
    if not axes:
        return
    fig = axes[0].figure

    def on_click(event):
        if event.button != 3:
            return
        
        ax = event.inaxes
        if ax in axes:
            elev, azim = ax.elev, ax.azim
            for other_ax in axes:
                if other_ax is not ax:
                    other_ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    return cid  # return connection id for optional later disconnect

def plot_pde_system(array, T=1, noise=0):
  half = int(array.shape[1] / 2)
  fig, ax = plt.subplots()
  noisedarray = array.copy()
  noisedarray[:, 1:, :] += np.random.normal(scale=noise, size=noisedarray[:, 1:, :].shape)
  
  @widgets.interact(j=(0, array.shape[0] - 1), t=(0, array.shape[1] - 1))
  def plot_io(j, t=20):
    print(j)
    ax.clear()
    ax.plot(array[j, 0, :], 'r', label="t=0")
    ax.plot(noisedarray[j, t, :], 'b', label=f"t={t / (array.shape[1] - 1) * T}")
    ax.set_title(j)
    ax.legend()

# assuming data is N x 2 array
def twodim_colors(data):
  hue_min, hue_max = 0, 0.7
  sat_min, sat_max = 0.3, 1 

  hue = np.interp(data[:, 0], (data[:, 0].min(), data[:, 0].max()), (hue_min, hue_max))
  sat = np.flip(np.interp(data[:, 1], (data[:, 1].min(), data[:, 1].max()), (sat_min, sat_max)))

  color_array = np.column_stack([hue, np.ones_like(hue), sat])

  return hsv_to_rgb(color_array)

def determine_params(paramarr):
  encoding_param = []
  P = paramarr.shape[1]

  if P == 1:
    return [0]

  for p in range(P):
    if np.abs(paramarr[0, p] - paramarr[1, p]) > 0:
      encoding_param.append(p)

  return encoding_param

def reset_model_weights(layer):
  if hasattr(layer, 'reset_parameters'):
    layer.reset_parameters()
  else:
    if hasattr(layer, 'children'):
      for child in layer.children():
        reset_model_weights(child)

def duplicate_rows(arr, T):
    N, P = arr.shape

    new_shape = (N * T, P)
    result_array = np.empty(new_shape, dtype=arr.dtype)
    
    for i in range(N):
        result_array[i * T : (i + 1) * T, :] = arr[i, :]

    return result_array

def reduce_params(params, tol=1e-4):
  arr = np.asarray(params)
  if arr.ndim != 2:
      raise ValueError("params must be 2-D (N × p)")

  # np.ptp gives max–min along each column
  varying = np.ptp(arr, axis=0) > tol
  out = arr[:, varying]
  return out


# =============================================================================
# Result Visualization Functions
# =============================================================================

def is_2d_spatial(data):
    """Check if data has 2D spatial dimensions."""
    if hasattr(data, 'data'):
        return len(data.data.shape) == 4
    return len(data.shape) == 4

# =============================================================================
# 1D Plotting Functions
# =============================================================================

def plot_data_1d(dset, samples=None, param_label="a", tmax=1.0, xmax=1.0):
    """
    Plot 1D spatiotemporal data as heatmaps for selected samples.

    Args:
        dset: Dataset with .data (N, T, X) and .params attributes
        samples: List of sample indices to plot (default: low/mid/high param values)
        param_label: Label for the parameter axis
        tmax: Maximum time value for axis labeling
        xmax: Maximum spatial value for axis labeling

    Returns:
        matplotlib figure
    """
    if samples is None:
        params = dset.params[:, 0] if len(dset.params.shape) > 1 else dset.params
        sorted_idx = np.argsort(params)
        samples = [sorted_idx[0], sorted_idx[len(sorted_idx)//2], sorted_idx[-1]]
    extent = [0, tmax, 0, xmax]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, s in zip(axes, samples):
        im = ax.imshow(dset.data[s].T, aspect='auto', origin='lower', cmap='viridis', extent=extent)
        ax.set_box_aspect(1)
        param_val = dset.params[s, 0] if len(dset.params.shape) > 1 else dset.params[s]
        ax.set_title(f"${param_label}$ = {param_val:.3f}")
        ax.set_xlabel("$t$"); ax.set_ylabel("$x$")
        plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    return fig

def plot_prediction_1d(ground_truth, prediction, sample_idx=0, title="", windows=1, tmax=1.0, xmax=1.0):
    """
    Plot 1D prediction comparison: ground truth, prediction, error, and final time slice.

    Args:
        ground_truth: Array of shape (N, T, X) with ground truth data
        prediction: Array of shape (N, T-1, X) with model predictions
        sample_idx: Index of sample to visualize
        title: Plot title
        windows: Number of windows (for drawing vertical lines)
        tmax: Maximum time value
        xmax: Maximum spatial value

    Returns:
        matplotlib figure
    """
    gt, pred = ground_truth[sample_idx, 1:], prediction[sample_idx]
    error = np.abs(gt - pred)
    extent = [0, tmax, 0, xmax]
    xs = np.linspace(0, xmax, gt.shape[1])
    window_times = [tmax * i / windows for i in range(1, windows)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, data, cmap, ttl in [(axes[0], gt, 'viridis', "Ground Truth"),
                                 (axes[1], pred, 'viridis', "Prediction"),
                                 (axes[2], error, 'hot', "Absolute Error")]:
        im = ax.imshow(data.T, aspect='auto', origin='lower', cmap=cmap, extent=extent)
        ax.set_title(ttl); ax.set_xlabel("$t$"); ax.set_ylabel("$x$")
        for wt in window_times: ax.axvline(x=wt, color='w', linestyle='-', linewidth=0.5)
        plt.colorbar(im, ax=ax)
    axes[3].plot(xs, gt[-1], 'b-', lw=1.25, label='Exact')
    axes[3].plot(xs, pred[-1], 'r--', lw=1.25, label='Predicted')
    axes[3].set_title(f"$t$ = {tmax:.2f}"); axes[3].set_xlabel("$x$"); axes[3].legend()
    plt.suptitle(title if title else f"Sample {sample_idx}")
    plt.tight_layout()
    return fig

def plot_compare_1d(ground_truth, predictions_dict, sample_idx=0, title="", tmax=1.0, xmax=1.0):
    """
    Compare multiple 1D models at selected time slices.

    Args:
        ground_truth: Array of shape (N, T, X)
        predictions_dict: Dict mapping model names to prediction arrays (N, T-1, X)
        sample_idx: Index of sample to visualize
        title: Plot title
        tmax: Maximum time value
        xmax: Maximum spatial value

    Returns:
        matplotlib figure
    """
    gt = ground_truth[sample_idx, 1:]
    T, X = gt.shape
    times_idx = [T // 3, 2 * T // 3, T - 1]
    times_phys = [tmax * (t + 1) / T for t in times_idx]
    xs = np.linspace(0, xmax, X)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = plt.cm.tab10.colors
    for col, (t_idx, t_phys) in enumerate(zip(times_idx, times_phys)):
        axes[0, col].plot(xs, gt[t_idx], 'k-', lw=1.25, label='Exact')
        for i, (name, pred) in enumerate(predictions_dict.items()):
            axes[0, col].plot(xs, pred[sample_idx, t_idx], '--', color=colors[i], lw=1.25, label=name)
        axes[0, col].set_title(f"$t$ = {t_phys:.2f}"); axes[0, col].set_xlabel("$x$"); axes[0, col].legend()
        for i, (name, pred) in enumerate(predictions_dict.items()):
            axes[1, col].plot(xs, np.abs(gt[t_idx] - pred[sample_idx, t_idx]), '-', color=colors[i], lw=1.25, label=name)
        axes[1, col].set_title(f"Error at $t$ = {t_phys:.2f}"); axes[1, col].set_xlabel("$x$")
        axes[1, col].set_yscale('log'); axes[1, col].legend()
        axes[1, col].set_ylim(bottom=max(axes[1, col].get_ylim()[0], 1e-6))
    plt.suptitle(title if title else f"Model Comparison - Sample {sample_idx}")
    plt.tight_layout()
    return fig

# =============================================================================
# 2D Plotting Functions
# =============================================================================

def plot_data_2d(dset, samples=None, param_label="a", n_times=4, n_samples=3,
                 tmax=1.0, xmax=1.0, ymax=1.0):
    """
    Plot 2D spatiotemporal data as image grids.

    Args:
        dset: Dataset with .data (N, T, X, Y) and .params attributes
        samples: List of sample indices (default: low/mid/high param values)
        param_label: Label for the parameter
        n_times: Number of time snapshots to show
        n_samples: Number of samples to show
        tmax, xmax, ymax: Physical domain extents

    Returns:
        matplotlib figure
    """
    import matplotlib.gridspec as gridspec

    N, T, X, Y = dset.data.shape
    if samples is None:
        params = dset.params[:, 0] if len(dset.params.shape) > 1 else dset.params
        sorted_idx = np.argsort(params)
        samples = [sorted_idx[int(i)] for i in np.linspace(0, len(sorted_idx)-1, n_samples)]
    time_indices = np.linspace(0, T-1, n_times, dtype=int)
    time_phys = [tmax * t / (T-1) for t in time_indices]
    extent = [0, xmax, 0, ymax]

    fig = plt.figure(figsize=(3.2*n_times + 0.8, 3*n_samples))
    gs = gridspec.GridSpec(n_samples, n_times + 1, width_ratios=[1]*n_times + [0.05], wspace=0.05, hspace=0.25)

    for row, s in enumerate(samples):
        param_val = dset.params[s, 0] if len(dset.params.shape) > 1 else dset.params[s]
        vmin, vmax = dset.data[s].min(), dset.data[s].max()

        for col, (t_idx, t_phys) in enumerate(zip(time_indices, time_phys)):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(dset.data[s, t_idx].T, aspect='equal', origin='lower',
                          cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
            if row == n_samples - 1: ax.set_xlabel("$x$")
            else: ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"${param_label}$={param_val*0.98:.2f}\n$y$")
            else: ax.set_yticklabels([])
            if row == 0: ax.set_title(f"$t$ = {t_phys:.2f}")

        cax = fig.add_subplot(gs[row, -1])
        fig.colorbar(im, cax=cax)

    return fig

def plot_prediction_2d(ground_truth, prediction, sample_idx=0, title="", n_times=4,
                       tmax=1.0, xmax=1.0, ymax=1.0):
    """
    Plot 2D prediction comparison: ground truth, prediction, and error at multiple times.

    Args:
        ground_truth: Array of shape (N, T, X, Y)
        prediction: Array of shape (N, T-1, X, Y)
        sample_idx: Index of sample to visualize
        title: Plot title
        n_times: Number of time snapshots
        tmax, xmax, ymax: Physical domain extents

    Returns:
        matplotlib figure
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm

    gt, pred = ground_truth[sample_idx, 1:], prediction[sample_idx]
    error = np.abs(gt - pred)
    T = gt.shape[0]
    time_indices = np.linspace(0, T-1, n_times, dtype=int)
    time_phys = [tmax * (t + 1) / T for t in time_indices]
    extent = [0, xmax, 0, ymax]
    vmin, vmax = min(gt.min(), pred.min()), max(gt.max(), pred.max())
    err_min, err_max = max(error.min(), 1e-10), error.max()

    fig = plt.figure(figsize=(3.2*n_times + 1.2, 9))
    gs = gridspec.GridSpec(3, n_times + 1, width_ratios=[1]*n_times + [0.05], wspace=0.05, hspace=0.2)

    row_data = [(gt, 'viridis', vmin, vmax, "Ground Truth", None),
                (pred, 'viridis', vmin, vmax, "Prediction", None),
                (error, 'jet', err_min, err_max, "Error (log)", LogNorm(vmin=err_min, vmax=err_max))]

    for row, (data, cmap, vlo, vhi, label, norm) in enumerate(row_data):
        for col, (t_idx, t_phys) in enumerate(zip(time_indices, time_phys)):
            ax = fig.add_subplot(gs[row, col])
            if norm: im = ax.imshow(data[t_idx].T, aspect='equal', origin='lower', cmap=cmap, extent=extent, norm=norm)
            else: im = ax.imshow(data[t_idx].T, aspect='equal', origin='lower', cmap=cmap, extent=extent, vmin=vlo, vmax=vhi)
            if row == 2: ax.set_xlabel("$x$")
            else: ax.set_xticklabels([])
            if col == 0: ax.set_ylabel(f"{label}\n$y$")
            else: ax.set_yticklabels([])
            if row == 0: ax.set_title(f"$t$ = {t_phys:.2f}")
        cax = fig.add_subplot(gs[row, -1])
        fig.colorbar(im, cax=cax)

    fig.suptitle(title if title else f"Sample {sample_idx}", fontsize=14)
    return fig

def plot_compare_2d(ground_truth, predictions_dict, sample_idx=0, title="",
                    tmax=1.0, xmax=1.0, ymax=1.0):
    """
    Compare multiple 2D models at final time with error visualization.

    Args:
        ground_truth: Array of shape (N, T, X, Y)
        predictions_dict: Dict mapping model names to prediction arrays
        sample_idx: Index of sample to visualize
        title: Plot title
        tmax, xmax, ymax: Physical domain extents

    Returns:
        matplotlib figure
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LogNorm

    initial = ground_truth[sample_idx, 0]
    gt = ground_truth[sample_idx, 1:]
    n_models = len(predictions_dict)
    n_cols = 1 + n_models
    extent = [0, xmax, 0, ymax]

    all_preds = [pred[sample_idx, -1] for pred in predictions_dict.values()]
    vmin = min(gt[-1].min(), min(p.min() for p in all_preds), initial.min())
    vmax = max(gt[-1].max(), max(p.max() for p in all_preds), initial.max())

    all_errors = [np.abs(gt[-1] - pred[sample_idx, -1]) for pred in predictions_dict.values()]
    err_min = max(min(e.min() for e in all_errors), 1e-10)
    err_max = max(e.max() for e in all_errors)

    fig = plt.figure(figsize=(3.2*n_cols + 1.2, 6.5))
    gs = gridspec.GridSpec(2, n_cols + 1, width_ratios=[1]*n_cols + [0.05], wspace=0.05, hspace=0.15)

    # Ground truth final
    ax = fig.add_subplot(gs[0, 0])
    im_val = ax.imshow(gt[-1].T, aspect='equal', origin='lower', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title("Ground Truth", fontweight='bold')
    ax.set_xticklabels([])
    ax.set_ylabel("$y$")

    # Initial condition
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(initial.T, aspect='equal', origin='lower', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title("Initial ($t$=0)", fontweight='bold')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    # Model predictions and errors
    for col, (name, pred) in enumerate(predictions_dict.items(), start=1):
        pred_final = pred[sample_idx, -1]
        error = np.abs(gt[-1] - pred_final)
        mae = np.mean(error)

        ax = fig.add_subplot(gs[0, col])
        ax.imshow(pred_final.T, aspect='equal', origin='lower', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\nMAE: {mae:.2e}", fontweight='bold')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax = fig.add_subplot(gs[1, col])
        im_err = ax.imshow(error.T, aspect='equal', origin='lower', cmap='jet', extent=extent,
                           norm=LogNorm(vmin=err_min, vmax=err_max))
        ax.set_title("Error (log)")
        ax.set_xlabel("$x$")
        ax.set_yticklabels([])

    cax_val = fig.add_subplot(gs[0, -1])
    fig.colorbar(im_val, cax=cax_val, label='u')
    cax_err = fig.add_subplot(gs[1, -1])
    fig.colorbar(im_err, cax=cax_err, label='|error|')

    fig.suptitle(title if title else f"Model Comparison - Sample {sample_idx}", fontsize=14)
    plt.tight_layout()
    return fig

# =============================================================================
# Auto-dispatch Functions
# =============================================================================

def plot_data(dset, samples=None, param_label="a", n_times=4, n_samples=3,
              tmax=1.0, xmax=1.0, ymax=1.0):
    """
    Plot dataset samples. Auto-dispatches to 1D or 2D version based on data shape.

    Args:
        dset: Dataset with .data and .params attributes
        samples: List of sample indices to plot
        param_label: Label for the parameter
        n_times: Number of time snapshots (2D only)
        n_samples: Number of samples (2D only)
        tmax, xmax, ymax: Physical domain extents

    Returns:
        matplotlib figure
    """
    if is_2d_spatial(dset):
        return plot_data_2d(dset, samples, param_label, n_times, n_samples, tmax, xmax, ymax)
    else:
        return plot_data_1d(dset, samples, param_label, tmax, xmax)

def plot_prediction(ground_truth, prediction, sample_idx=0, title="", windows=1,
                    n_times=4, tmax=1.0, xmax=1.0, ymax=1.0):
    """
    Plot model prediction vs ground truth. Auto-dispatches to 1D or 2D version.

    Args:
        ground_truth: Ground truth array (N, T, ..spatial dims..)
        prediction: Prediction array (N, T-1, ..spatial dims..)
        sample_idx: Index of sample to visualize
        title: Plot title
        windows: Number of windows for vertical lines (1D only)
        n_times: Number of time snapshots (2D only)
        tmax, xmax, ymax: Physical domain extents

    Returns:
        matplotlib figure
    """
    if is_2d_spatial(ground_truth):
        return plot_prediction_2d(ground_truth, prediction, sample_idx, title, n_times, tmax, xmax, ymax)
    else:
        return plot_prediction_1d(ground_truth, prediction, sample_idx, title, windows, tmax, xmax)

def plot_compare(ground_truth, predictions_dict, sample_idx=0, title="",
                 tmax=1.0, xmax=1.0, ymax=1.0):
    """
    Compare multiple models. Auto-dispatches to 1D or 2D version.

    Args:
        ground_truth: Ground truth array (N, T, ..spatial dims..)
        predictions_dict: Dict mapping model names to prediction arrays
        sample_idx: Index of sample to visualize
        title: Plot title
        tmax, xmax, ymax: Physical domain extents

    Returns:
        matplotlib figure
    """
    if is_2d_spatial(ground_truth):
        return plot_compare_2d(ground_truth, predictions_dict, sample_idx, title, tmax, xmax, ymax)
    else:
        return plot_compare_1d(ground_truth, predictions_dict, sample_idx, title, tmax, xmax)
