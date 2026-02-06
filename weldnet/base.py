# Base networks and utility functions

import time
import glob
import itertools
import datetime
import copy
import os
import pickle
import random
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import seaborn as sns
import deepxde as dde
import matplotlib.cm as cm
import ruptures as rpt
from concurrent.futures import ThreadPoolExecutor
from typing import Union

from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from copy import deepcopy

from omegaconf import DictConfig, OmegaConf, ListConfig

from .config import create_object, load_config


# =============================================================================
# Utility functions from models2.py (replaces jcmodels/)
# =============================================================================

def prod(l: list, to_dtype=int):
    """Product of list elements."""
    if len(l) > 0:
        p = 1.0
        for ele in l:
            p *= ele
        return to_dtype(p)
    else:
        return 0


def conv_out_dim(input_dim, kernel_size, stride, padding):
    """Calculate output dimension after convolution."""
    if isinstance(input_dim, list) or isinstance(input_dim, ListConfig):
        out_dim = [conv_out_dim(in_dim, kernel_size, stride, padding) for in_dim in input_dim]
    else:
        out_dim = math.floor((input_dim - kernel_size + 2 * padding) / stride) + 1
    return out_dim


def tconv_out_dim(input_dim, kernel_size, stride, padding, output_padding=0):
    """Calculate output dimension after transposed convolution."""
    if isinstance(input_dim, list) or isinstance(input_dim, ListConfig):
        if not (isinstance(output_padding, list) or isinstance(output_padding, ListConfig)):
            output_padding = [output_padding] * len(input_dim)
        assert len(output_padding) == len(input_dim)
        out_dim = [tconv_out_dim(in_dim, kernel_size, stride, padding, out_padding)
                   for in_dim, out_padding in zip(input_dim, output_padding)]
    else:
        out_dim = (input_dim - 1) * stride - 2 * padding + kernel_size + output_padding
    return out_dim


# =============================================================================
# Coder base classes from models2.py
# =============================================================================

class Coder(nn.Module):
    """Base class for encoders and decoders."""

    def configure_dims(self, dims: Union[int, list]):
        if not (isinstance(dims, list) or isinstance(dims, ListConfig)):
            dims = [dims]
        self.dim = len(dims)
        return dims

    def forward(self, x, t=None):
        if t is None:
            return self.blocks(x)
        else:
            x = torch.cat([x, t], dim=-1)
            return self.blocks(x)


class ConvEncoder(Coder):
    """Convolutional encoder."""

    def __init__(self, in_dims, latents_dims, out_channels, kernel_stride_paddings,
                 actvn=nn.ReLU(), in_channel=1, padding_mode="zeros", mlp=False, **kwargs):
        super().__init__()
        in_dims = self.configure_dims(in_dims)
        self.configure_conv_dims(in_dims, kernel_stride_paddings)

        conv = nn.Conv2d if self.dim == 2 else nn.Conv1d
        out_channels.insert(0, in_channel)

        if mlp:
            self.blocks = nn.Sequential(
                *[nn.Sequential(
                    conv(out_channels[i], out_channels[i+1], ksp[0], ksp[1], ksp[2], padding_mode=padding_mode),
                    actvn
                ) for i, ksp in enumerate(kernel_stride_paddings)],
                nn.Flatten(),
                nn.Linear(out_channels[-1] * prod(self.conv_layers_config[-1]["out_dims"]), 256 if self.dim == 1 else 512),
                actvn,
                nn.Linear(256 if self.dim == 1 else 512, latents_dims)
            )
        else:
            self.blocks = nn.Sequential(
                *[nn.Sequential(
                    conv(out_channels[i], out_channels[i+1], ksp[0], ksp[1], ksp[2], padding_mode=padding_mode),
                    actvn
                ) for i, ksp in enumerate(kernel_stride_paddings)],
                nn.Flatten(),
                nn.Linear(out_channels[-1] * prod(self.conv_layers_config[-1]["out_dims"]), latents_dims),
            )

    def configure_conv_dims(self, in_dims, kernel_stride_paddings):
        self.conv_layers_config = []
        for i, ksp in enumerate(kernel_stride_paddings):
            in_dims = in_dims if i == 0 else self.conv_layers_config[-1]["out_dims"]
            self.conv_layers_config.append(dict(
                in_dims=in_dims,
                out_dims=conv_out_dim(in_dims, ksp[0], ksp[1], ksp[2]),
                kernel_size=ksp[0],
                stride=ksp[1],
                padding=ksp[2],
            ))


class ConvDecoder(Coder):
    """Convolutional decoder (transpose conv)."""

    def __init__(self, out_dims, latents_dims, in_channels, kernel_stride_paddings,
                 actvn=nn.ReLU(), out_channel=1, uncond=True, padding_mode="zeros", mlp=False, **kwargs):
        super().__init__()
        if not uncond:
            latents_dims += 1
        out_dims = self.configure_dims(out_dims)
        outpaddings = self.configure_tconv_dims(out_dims, kernel_stride_paddings)

        tconv = nn.ConvTranspose2d if self.dim == 2 else nn.ConvTranspose1d
        in_channels.append(out_channel)

        if mlp:
            self.blocks = nn.Sequential(
                nn.Linear(latents_dims, 256 if self.dim == 1 else 512),
                actvn,
                nn.Linear(256 if self.dim == 1 else 512, in_channels[0] * prod(self.tconv_layers_config[0]["in_dims"])),
                nn.Unflatten(1, (in_channels[0], *self.tconv_layers_config[0]["in_dims"])),
                *[nn.Sequential(
                    actvn,
                    tconv(in_channels[i], in_channels[i+1], ksp[0], ksp[1], ksp[2], outpadding, padding_mode=padding_mode)
                ) for i, (ksp, outpadding) in enumerate(zip(kernel_stride_paddings, outpaddings))]
            )
        else:
            self.blocks = nn.Sequential(
                nn.Linear(latents_dims, in_channels[0] * prod(self.tconv_layers_config[0]["in_dims"])),
                nn.Unflatten(1, (in_channels[0], *self.tconv_layers_config[0]["in_dims"])),
                *[nn.Sequential(
                    actvn,
                    tconv(in_channels[i], in_channels[i+1], ksp[0], ksp[1], ksp[2], outpadding, padding_mode=padding_mode)
                ) for i, (ksp, outpadding) in enumerate(zip(kernel_stride_paddings, outpaddings))]
            )

    def configure_tconv_dims(self, out_dims, kernel_stride_paddings):
        self.tconv_layers_config = []
        for i, ksp in enumerate(kernel_stride_paddings[::-1]):
            out_dims = out_dims if i == 0 else self.tconv_layers_config[0]["in_dims"]
            self.tconv_layers_config.insert(0, dict(
                in_dims=conv_out_dim(out_dims, ksp[0], ksp[1], ksp[2]),
                out_dims=out_dims,
                kernel_size=ksp[0],
                stride=ksp[1],
                padding=ksp[2],
            ))
        outpaddings = self.get_outpaddings()
        return outpaddings

    def get_outpaddings(self):
        outpaddings = []
        for config in self.tconv_layers_config:
            outpaddings.append([out_dim - out_dim_no_padding
                               for (out_dim, out_dim_no_padding) in
                               zip(config["out_dims"], tconv_out_dim(config["in_dims"], config["kernel_size"], config["stride"], config["padding"]))])
        return outpaddings


class LSTMEncoder(Coder):
    """LSTM-based encoder."""

    def __init__(self, in_dims, latents_dims, num_layers=1, **kwargs):
        super().__init__()
        in_dims = self.configure_dims(in_dims)
        self.latents_dims, self.num_layers = latents_dims, num_layers
        self.lstm = nn.LSTM(
            prod(in_dims),
            latents_dims,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            proj_size=0,
            device=None)

    def _forward_lstm(self, x):
        hidden_state = []
        prev_h = torch.rand(self.num_layers, x.shape[0], self.latents_dims)
        prev_c = torch.rand(self.num_layers, x.shape[0], self.latents_dims)
        for s in range(x.shape[1]):
            out, (prev_h, prev_c) = self.lstm(x[:, s:s+1, :], (prev_h, prev_c))
            hidden_state.append(prev_h[-1:, :, :])
        return torch.cat(hidden_state, dim=0).transpose(0, 1)

    def forward(self, x):
        x = x.view(*x.shape[:2], -1)
        x = self._forward_lstm(x)
        return x


class LSTMDecoder(Coder):
    """LSTM-based decoder."""

    def __init__(self, out_dims, latents_dims, num_layers=1, **kwargs):
        super().__init__()
        out_dims = self.configure_dims(out_dims)
        self.num_layers, self.out_dims = num_layers, out_dims
        self.lstm = nn.LSTM(
            latents_dims,
            prod(out_dims),
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            proj_size=0,
            device=None)

    def _forward_lstm(self, x):
        hidden_state = []
        prev_h = torch.rand(self.num_layers, x.shape[0], prod(self.out_dims))
        prev_c = torch.rand(self.num_layers, x.shape[0], prod(self.out_dims))
        for s in range(x.shape[1]):
            out, (prev_h, prev_c) = self.lstm(x[:, s:s+1, :], (prev_h, prev_c))
            hidden_state.append(prev_h[-1:, :, :])
        return torch.cat(hidden_state, dim=0).transpose(0, 1)

    def forward(self, x):
        x = self._forward_lstm(x)
        x = x.view(*x.shape[:2], *self.out_dims)
        return x


# =============================================================================
# AutoEncoder classes from models2.py
# =============================================================================

class AutoEncoder(nn.Module):
    """Base autoencoder class with encode/decode logic."""

    def encode(self, x):
        if self.dim == 1:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
                S = x.shape[1]
                return self.encoder(x.reshape(-1, 1, x.shape[-1])).reshape(S, -1)
            else:
                B = x.shape[0]
                S = x.shape[1]
                return self.encoder(x.reshape(-1, 1, x.shape[-1])).reshape(B, S, -1)
        else:
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
                B, S, h, w = x.shape
                out = self.encoder(x.reshape(B * S, 1, h, w))
                return out.reshape(S, -1)
            else:
                B, S, h, w = x.shape
                out = self.encoder(x.reshape(B * S, 1, h, w))
                return out.reshape(B, S, -1)

    def decode(self, x, t=None):
        if self.dim == 1:
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
                S = x.shape[1]
                return self.decoder(x.reshape(-1, x.shape[-1]), t).reshape(S, -1)
            else:
                B = x.shape[0]
                S = x.shape[1]
                return self.decoder(x.reshape(-1, x.shape[-1]), t).reshape(B, S, -1)
        else:
            if len(x.shape) == 2:
                S = x.shape[0]
                out = self.decoder(x.reshape(-1, x.shape[-1]), t)
                h_out, w_out = out.shape[-2:]
                return out.reshape(S, h_out, w_out)
            else:
                B = x.shape[0]
                S = x.shape[1]
                out = self.decoder(x.reshape(-1, x.shape[-1]), t)
                h_out, w_out = out.shape[-2:]
                return out.reshape(B, S, h_out, w_out)

    def forward(self, x, t=None):
        return self.decode(self.encode(x), t)


class FCAutoEncoder(AutoEncoder):
    """Fully connected autoencoder (placeholder for compatibility)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        # Minimal implementation for compatibility
        self.dim = 1
        self.pcadim = 0


class ConvAutoEncoder(AutoEncoder):
    """Convolutional autoencoder."""

    def __init__(self, config: DictConfig):
        super().__init__()
        in_dims = config.sample.spatial_resolution
        latents_dims = config.sample.latents_dims
        out_channels = config.downblocks.channels
        kernel_stride_paddings = config.downblocks.kernel_stride_paddings
        actvn = get_activation(config.downblocks.actvn)
        padding_mode = config.downblocks.get("padding_mode", 'zeros')
        self.dim = 1 if not (isinstance(in_dims, list) or isinstance(in_dims, ListConfig)) else len(in_dims)

        mlp = config.get("mlp", True)

        self.encoder = ConvEncoder(
            in_dims=in_dims,
            latents_dims=latents_dims,
            out_channels=out_channels,
            kernel_stride_paddings=kernel_stride_paddings,
            actvn=actvn,
            padding_mode=padding_mode,
            mlp=mlp
        )
        self.decoder = ConvDecoder(
            out_dims=in_dims,
            latents_dims=latents_dims,
            in_channels=out_channels[::-1],
            kernel_stride_paddings=kernel_stride_paddings[::-1],
            actvn=actvn,
            padding_mode=padding_mode,
            mlp=mlp
        )

        self.pcadim = 0


class ConvAutoEncoder2D(ConvAutoEncoder):
    """2D Convolutional autoencoder (reshapes 1D input to 2D)."""

    def __init__(self, config: DictConfig):
        from math import sqrt
        in_dims = config.sample.spatial_resolution
        config.sample.spatial_resolution = [int(sqrt(in_dims)), int(sqrt(in_dims))] if not (isinstance(in_dims, list) or isinstance(in_dims, ListConfig)) else in_dims
        self.dim = 1 if not (isinstance(in_dims, list) or isinstance(in_dims, ListConfig)) else len(in_dims)
        super().__init__(config)

    def hacky_unflatten(self, x):
        from math import sqrt
        print(x.shape)
        return x.reshape(*x.shape[:-1], int(sqrt(x.shape[-1])), int(sqrt(x.shape[-1])))

    def hacky_flatten(self, x):
        print(x.shape)
        return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])

    def encode(self, x):
        return super().encode(x)

    def decode(self, x, t=None):
        return super().decode(x, t)


class LSTMAutoEncoder(AutoEncoder):
    """LSTM-based autoencoder."""

    def __init__(self, config: DictConfig):
        super().__init__()
        in_dims = config.sample.spatial_resolution
        latents_dims = config.sample.latents_dims
        self.dim = 1
        self.encoder = LSTMEncoder(
            in_dims=in_dims,
            latents_dims=latents_dims,
        )
        self.decoder = LSTMDecoder(
            out_dims=in_dims,
            latents_dims=latents_dims,
        )

    def encode(self, x):
        if len(x.shape) == 2:
            x = x.view(-1, *x.shape)
        return self.encoder(x)

    def decode(self, x, t=None):
        out = self.decoder(x)
        if out.shape[0] == 1:
            out = out.view(*out.shape[1:])
        return out

    def forward(self, x, t=None):
        return self.decode(self.encode(x))


class ResnetAutoEncoder(AutoEncoder):
    """ResNet-based autoencoder (placeholder for compatibility)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.dim = 1
        self.pcadim = 0


class ConvLSTMAutoEncoder(AutoEncoder):
    """Conv-LSTM autoencoder (placeholder for compatibility)."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.dim = 1
        self.pcadim = 0


# List of JC modules (now defined locally, replacing jcmodels.networks import)
JC_Modules = [FCAutoEncoder, ConvAutoEncoder, ConvAutoEncoder2D, ResnetAutoEncoder, LSTMAutoEncoder, ConvEncoder, ConvLSTMAutoEncoder]

plt.rcParams["figure.figsize"] = (7, 3)

BASEDIR = "savedmodels/ae"


def get_activation(activation):
    """Get activation function by name string or return if already an activation."""
    if not isinstance(activation, str):
        return activation

    act_lower = activation.lower()
    if act_lower == "relu":
        return nn.ReLU()
    elif act_lower == "tanh":
        return nn.Tanh()
    elif act_lower == "sigmoid":
        return nn.Sigmoid()
    elif act_lower == "gelu":
        return nn.GELU()
    elif act_lower == "leakyrelu":
        return nn.LeakyReLU()
    elif act_lower == "silu" or act_lower == "swish":
        return nn.SiLU()
    else:
        return nn.ReLU()


# Alias for backward compatibility
get_actvn = get_activation


def train_pca_shared(data, n_components, datadim=1):
    """
    Shared PCA training utility function.

    Args:
        data: Input tensor to fit PCA on
        n_components: Number of PCA components
        datadim: Number of trailing dimensions to flatten

    Returns:
        pca_dict: Dictionary with 'tensor' and 'center' keys
        errors: Tuple of (L1, L2, Linf) relative projection errors
    """
    data = data.reshape([-1] + list(data.shape[-datadim:]))

    pca = PCA(n_components=n_components)
    pca = pca.fit(data.cpu().detach().numpy() if hasattr(data, 'cpu') else data)

    pca_dict = {
        "tensor": torch.tensor(pca.components_, dtype=torch.float32),
        "center": torch.tensor(pca.mean_, dtype=torch.float32)
    }

    # Compute projection errors
    if hasattr(data, 'cpu'):
        data_np = data
    else:
        data_np = torch.tensor(data.copy() if isinstance(data, np.ndarray) else data, dtype=torch.float32)

    encdata = torch.matmul(data_np - pca_dict["center"], pca_dict["tensor"].T)
    projdata = torch.matmul(encdata, pca_dict["tensor"]) + pca_dict["center"]

    errors = tuple(
        torch.linalg.vector_norm(data_np - projdata, ord=ord) / torch.linalg.vector_norm(data_np, ord=ord)
        for ord in [1, 2, torch.inf]
    )

    return pca_dict, errors


class PCAMixin:
    """Mixin providing train_pca functionality for models with pcadim and datadim attributes."""

    def train_pca(self, data):
        """Train PCA on data and store in self.pca."""
        self.pca, errors = train_pca_shared(data, self.pcadim, getattr(self, 'datadim', 1))
        return errors


def compute_relative_errors(orig, out, ords=(2,), aggregate=True, times=None):
    """
    Compute relative errors between original and output arrays.

    Args:
        orig: Original array (N, T, ...) or (N, ...)
        out: Output array, same shape as orig
        ords: Tuple of norm orders to compute (e.g., (1, 2, np.inf))
        aggregate: If True, flatten and return mean errors. If False, return per-sample errors.
        times: Optional list of time indices to compute errors for (only used when aggregate=False)

    Returns:
        If aggregate=True: tuple of mean relative errors for each ord
        If aggregate=False: list of per-time errors or array of per-sample errors
    """
    n = orig.shape[0]

    if aggregate:
        orig_flat = orig.reshape([n, -1])
        out_flat = out.reshape([n, -1])
        testerrs = []
        for o in ords:
            testerrs.append(np.mean(np.linalg.norm(orig_flat - out_flat, axis=1, ord=o) /
                                    np.linalg.norm(orig_flat, axis=1, ord=o)))
        return tuple(testerrs)
    else:
        o = ords[0]
        if times is not None and len(times) == 1:
            t = times[0]
            origslice = orig[:, t].reshape([n, -1])
            outslice = out[:, t].reshape([n, -1])
            return np.linalg.norm(origslice - outslice, axis=1, ord=o) / np.linalg.norm(origslice, axis=1, ord=o)
        else:
            testerrs = []
            for t in range(orig.shape[1]):
                origslice = orig[:, t].reshape([n, -1])
                outslice = out[:, t].reshape([n, -1])
                testerrs.append(np.mean(np.linalg.norm(origslice - outslice, axis=1, ord=o) /
                                        np.linalg.norm(origslice, axis=1, ord=o)))
            return testerrs


def plot_errorparams_shared(net, get_operrs_func, param=-1):
    """
    Shared implementation for plotting error vs parameters.

    Args:
        net: The network object with dataset attribute
        get_operrs_func: Function that takes (net, times, testonly) and returns errors
        param: Parameter index to plot against, or -1 to auto-detect, or list of 2 for 3D plot
    """
    import matplotlib.pyplot as plt

    if param == -1:
        # Auto-detect one varying parameter
        param = 0
        P = net.dataset.params.shape[1]
        for p in range(P):
            if np.abs(net.dataset.params[0, p] - net.dataset.params[1, p]) > 0:
                param = p
                break

    l2error = np.asarray(get_operrs_func(net, times=[net.T - 1], testonly=False))
    params = net.dataset.params

    print(params.shape, l2error.shape)

    if isinstance(param, (list, tuple, np.ndarray)) and len(param) == 2:
        # 3D scatter plot for 2 varying parameters
        x = params[:, param[0]]
        y = params[:, param[1]]
        z = l2error

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=10)

        ax.set_xlabel(f"Param {param[0]}")
        ax.set_ylabel(f"Param {param[1]}")
        ax.set_zlabel("Operator Error")
        fig.colorbar(sc, ax=ax, label="Operator Error")
    else:
        # Fallback to 2D scatter if param is 1D
        fig, ax = plt.subplots()
        ax.scatter(params[:, param], l2error, s=2)
        ax.set_xlabel(f"Parameter {param}")
        ax.set_ylabel("Operator Error")

    fig.tight_layout()


def _get_helper_for_model(model):
    """Get the appropriate helper class for a model instance."""
    # Lazy imports to avoid circular dependencies
    from .othermodels import TimeInputModel, HighDimProp
    from .othermodels import TimeInputHelper, HighDimPropHelper
    from .weld import WeldNet, WeldHelper
    from .othermodels import LDNet, LDHelper
    from .othermodels import LDON, LDONHelper

    model_to_helper = {
        TimeInputModel: TimeInputHelper,
        HighDimProp: HighDimPropHelper,
        WeldNet: WeldHelper,
        LDNet: LDHelper,
        LDON: LDONHelper,
    }

    for model_cls, helper_cls in model_to_helper.items():
        if isinstance(model, model_cls):
            return helper_cls

    raise ValueError(f"No helper found for model type {type(model).__name__}")


def compare_operrs(models, labels=None, get_operrs_func=None):
    """
    Compare operator errors across multiple models.

    Args:
        models: List of model objects to compare
        labels: Optional list of labels for each model (defaults to indices)
        get_operrs_func: Optional function that takes (model, testonly) and returns errors.
                         If not provided, auto-detects based on model type.

    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if labels is None:
        labels = range(len(models))

    for lbl, model in zip(labels, models):
        if get_operrs_func is not None:
            operrs = get_operrs_func(model, testonly=True)
        else:
            helper = _get_helper_for_model(model)
            operrs = helper.get_operrs(model, testonly=True)
        ax.plot(np.log10(operrs), label=lbl)

    ax.legend()
    ax.set_ylabel(r"$\log_{10}$(Operator Error)")
    ax.set_xlabel("Time")

    fig.tight_layout()
    return fig


class Comparisons:
    """Unified interface for plotting/comparing any model type.

    All methods accept either a single model or a list of models.
    Single model: plots for that model. List: compares across models.
    """

    @staticmethod
    def get_operrs(model, times=None, testonly=True):
        helper = _get_helper_for_model(model)
        return np.asarray(helper.get_operrs(model, times=times, testonly=testonly))

    @staticmethod
    def plot_operrs(models, labels=None):
        """Plot operator errors over time. Single model or list."""
        if not isinstance(models, (list, tuple)):
            models = [models]
        return compare_operrs(models, labels=labels)

    @staticmethod
    def plot_op_predicts(models, **kwargs):
        """Interactive prediction plot. Single model or list."""
        if not isinstance(models, (list, tuple)):
            helper = _get_helper_for_model(models)
            return helper.plot_op_predicts(models, **kwargs)

        for model in models:
            helper = _get_helper_for_model(model)
            helper.plot_op_predicts(model, **kwargs)

    @staticmethod
    def plot_errorparams(models, labels=None, param=-1, ytop=None):
        """Plot error vs parameter. Single model or list."""
        import matplotlib.pyplot as plt

        if not isinstance(models, (list, tuple)):
            # Single model — use the shared single-model plot
            helper = _get_helper_for_model(models)
            return plot_errorparams_shared(models, helper.get_operrs, param)

        # Multiple models — comparison scatter
        markers = ('.', 'x', '^', '*', '1', '2', 's', 'o')
        params = models[0].dataset.params

        if param == -1:
            param = 0
            P = params.shape[1]
            for p in range(P):
                if np.abs(params[0, p] - params[1, p]) > 0:
                    param = p
                    break

        if labels is None:
            labels = [f"Model {i}" for i in range(len(models))]

        fig, ax = plt.subplots(figsize=(6, 4))

        for i, model in enumerate(models):
            helper = _get_helper_for_model(model)
            errors = np.asarray(helper.get_operrs(model, times=[model.T - 1], testonly=False))
            model_params = model.dataset.params

            marker = markers[i % len(markers)]
            ax.scatter(model_params[:, param], errors, s=5, label=labels[i], marker=marker)

        ax.set_xlabel(f"Parameter {param}")
        ax.set_ylabel("Final Time Operator Error")
        ax.set_yscale("log")

        if ytop is not None:
            ax.set_ylim(top=ytop)

        lgnd = ax.legend(loc="upper left")
        for handle in lgnd.legend_handles:
            handle.set_sizes([15])

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_projops(model, **kwargs):
        """Plot projection + operator errors (WeldNet only)."""
        from .weld import WeldNet, WeldHelper
        if not isinstance(model, WeldNet):
            raise TypeError(f"plot_projops is only supported for WeldNet models, got {type(model).__name__}")
        return WeldHelper.plot_projops(model, **kwargs)


class BaseHelper:
    """Base class for Helper classes with shared functionality."""

    def __init__(self, config):
        self.update_config(config)

    def update_config(self, config):
        from copy import deepcopy
        self.config = deepcopy(config)


def load_model_by_metadata(search_path, metadata, min_epochs=0, verbose=False):
    """
    Search for and load a model file matching the given metadata.

    Args:
        search_path: Glob pattern for model files (e.g., "savedmodels/ldnet/prefix*.pickle")
        metadata: Dictionary of metadata to match against
        min_epochs: Minimum number of training epochs required
        verbose: If True, print detailed matching information

    Returns:
        Tuple of (matched_dict, addr) if found, or (None, None) if not found
    """
    matching_files = glob.glob(search_path)

    print("Searching for model files matching prefix:", search_path)

    for addr in matching_files:
        try:
            with open(addr, "rb") as handle:
                dic = pickle.load(handle)
        except Exception as e:
            if verbose:
                print(f"Skipping {addr} due to read error: {e}")
            continue

        meta = dic.get("metadata", {})
        is_match = all(
            meta.get(k) == metadata.get(k)
            for k in metadata.keys()
        )

        # Check epoch requirements
        model_epochs = dic.get("epochs")
        if model_epochs is None:
            if verbose:
                print(f"Skipping {addr} due to missing epoch metadata.")
            continue
        elif isinstance(model_epochs, list):
            if sum(model_epochs) < min_epochs:
                if verbose:
                    print(f"Skipping {addr} due to insufficient epochs ({sum(model_epochs)} < {min_epochs})")
                continue
        elif model_epochs < min_epochs:
            if verbose:
                print(f"Skipping {addr} due to insufficient epochs ({model_epochs} < {min_epochs})")
            continue

        if is_match:
            print("Model match found. Loading from:", addr)
            return dic, addr
        elif verbose:
            print("Metadata mismatch in file:", addr)
            for k in metadata:
                print(f"{k}: saved={meta.get(k)} vs current={metadata.get(k)}")

    print("Load failed. No matching models found.")
    print("Searched:", matching_files)
    return None, None


def determine_param(dataset, encoding_param):
    if encoding_param == -1:
        encoding_param = []
        P = dataset.params.shape[1]

        for p in range(P):
            if np.abs(dataset.params[0, p] - dataset.params[1, p]) > 0:
                encoding_param.append(p)

    return encoding_param

class FFNet(nn.Module):
    def __init__(self, seq, activation):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(seq[i], seq[i+1]) for i in range(len(seq) - 1)])

        if type(activation) == type(""):
            activation = get_activation(activation)

        self.s = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if not i == len(self.layers) - 1:
                x = self.s(x)
        return x


class DeepONet(nn.Module):
    def __init__(self, branchseq, trunkseq, activation):
        super().__init__()

        self.branchnet = nn.ModuleList([nn.Linear(branchseq[i], branchseq[i+1]) for i in range(len(branchseq) - 1)])
        self.trunknet = nn.ModuleList([nn.Linear(trunkseq[i], trunkseq[i+1]) for i in range(len(trunkseq) - 1)])
        self.s = activation

    def forward(self, u, x):
        for i, layer in enumerate(self.branchnet):
            u = layer(u)
            if not i == len(self.branchnet) - 1:
                u = self.s(u)

        for i, layer in enumerate(self.trunknet):
            x = layer(x)
            if not i == len(self.trunknet) - 1:
                x = self.s(x)

        out = torch.einsum("nmk,tk->nmt", x, u).transpose(0, 2)
        return out


# Abstract class
class EncoderNet(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x):
        raise NotImplementedError()

    def decode(self, v):
        raise NotImplementedError()

    def right(self, v):
        raise NotImplementedError()

    def forward(self, x):
        v = self.encode(x)
        out = self.right(v)
        return out

    def project(self, x):
        v = self.encode(x)
        out = self.decode(v)
        return out

    def setup_data(self, dataset, trainnum, batch_size, t0=0, t1=1, loss=nn.MSELoss()):
        raise NotImplementedError

    def setup_trainloader(self, trainnum, batch_size, t0=0, t1=-1):
        raise NotImplementedError

    def setup_misc(self, loss=nn.MSELoss()):
        self.loss = loss

    def setup_models(self, **args):
        raise NotImplementedError

    def train_encoder(self, epochs, **args):
        raise NotImplementedError

    def plot_encoding(self, name, ep):
        assert(self.writer)

        half = int(self.testarray.shape[1] / 2)
        input = torch.tensor(self.testarray[:, :half]).to(self.device)

        test = self.encode(input)
        points = test.cpu().detach().numpy()

        for i in range(self.dataset.params.shape[1]):
            fig = plt.figure(figsize=(16, 4))
            fig.suptitle(f"Param {i+1}")

            n = points.shape[0]

            if points.shape[1] == 1:
                ax = fig.add_subplot()
                sns.kdeplot(points, fill=True, ax=ax)
                ax.set_ylabel("Probability")
            elif points.shape[1] == 2:
                ax = fig.add_subplot()
                sc = ax.scatter(points[:, 0], points[:, 1], c=self.dataset.params[-n:, i])
                colorbar = plt.colorbar(sc, ax=ax)
            else:
                m = points.shape[1]
                combinations_2 = list(combinations(range(m), 2))
                num = len(combinations_2)

                for idx, (col1, col2) in enumerate(combinations_2, start=1):
                    if idx > 6:
                        continue

                    ax = fig.add_subplot(1, min(6, num), idx)
                    ax.scatter(points[:, col1], points[:, col2], c=self.dataset.params[-n:, i])
                    ax.set_xlabel(f'{col1}')
                    ax.set_ylabel(f'{col2}')
                    ax.set_title(f'Scatter Plot {col1} vs {col2}')

            self.writer.add_figure(f'test/{name}-{i}', fig, global_step=ep)
            self.writer.flush()

        plt.close('all')

    def train_right(self):
        raise NotImplementedError()

    def get_projerr(self, ord=2):
        half = int(self.testarray.shape[1] // 2)

        domain = torch.tensor(self.testarray[:, :half]).to(self.device)
        projected = self.project(domain).cpu().detach().numpy()
        domain = domain.cpu().detach().numpy()
        return np.mean(np.linalg.norm(projected - domain, axis=1, ord=ord) / np.linalg.norm(domain, axis=1, ord=ord))

    def get_operr(self, arr=None, ord=2):
        if arr is None:
            arr = self.testarray

        half = int(arr.shape[1] // 2)

        domain = torch.tensor(arr[:, :half]).to(self.device)
        rangee = arr[:, half:]
        operator = self.forward(domain).cpu().detach().numpy()
        return np.mean(np.linalg.norm(operator - rangee, axis=1, ord=ord) / np.linalg.norm(rangee, axis=1, ord=ord))

    def get_generr(self, arr=None):
        if arr is None:
            arr = self.testarray

        half = int(arr.shape[1] // 2)

        domain = torch.tensor(arr[:, :half]).to(next(self.parameters()).get_device())
        rangee = arr[:, half:]
        operator = self.forward(domain).cpu().detach().numpy()
        return np.mean((np.linalg.norm(operator - rangee, axis=1)) ** 2) / (arr.shape[1] - 1)  # divide by the coefficient


class PCAAutoencoder(nn.Module):
    def __init__(self, inputdim, reduced, datadim=1):
        super().__init__()

        self.pcadim = 0
        self.inputdim = inputdim
        self.reduced = reduced
        self.datadim = datadim

        self.pcaTensor = torch.zeros((inputdim, reduced))
        self.pcaCenter = torch.zeros((inputdim))

    def train_pca(self, data):
        data = data.reshape([-1] + list(data.shape[-self.datadim:]))

        pca = PCA(n_components=self.reduced)
        pca = pca.fit(data)

        self.pcaTensor = torch.tensor(pca.components_, dtype=torch.float32)
        self.pcaCenter = torch.tensor(pca.mean_, dtype=torch.float32)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, enc):
        preshape = list(enc.shape[:-self.datadim])
        enc = enc.reshape([-1] + list(enc.shape[-self.datadim:]))

        self.pcaCenter = self.pcaCenter.to(enc.device)
        self.pcaTensor = self.pcaTensor.to(enc.device)

        out = torch.matmul(enc - self.pcaCenter, self.pcaTensor.T)
        return out.reshape(preshape + [-1])

    def decode(self, dec):
        dec = torch.tensor(dec, device=self.pcaTensor.device, dtype=torch.float32)
        dec = torch.matmul(dec, self.pcaTensor) + self.pcaCenter
        dec = dec.reshape([-1] + list(dec.shape[-self.datadim:]))

        return dec


class FFAutoencoder(nn.Module):
    def __init__(self, encodeSeq, decodeSeq, activation, datadim=1, pcadim=0):
        super().__init__()

        self.pcadim = pcadim
        self.pca = False

        if self.pcadim > 0:
            encodeSeq[0] = self.pcadim
            decodeSeq[-1] = self.pcadim

        self.encoder = FFNet(activation=activation, seq=encodeSeq)
        self.decoder = FFNet(activation=activation, seq=decodeSeq)
        self.s = activation

        self.reduced = encodeSeq[-1]
        self.datadim = datadim

    def train_pca(self, data):
        """Train PCA on data using shared utility."""
        print(data.shape)
        self.pca, errors = train_pca_shared(data, self.pcadim, datadim=1)
        return errors

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, enc):
        if self.datadim == 2:
            enc = enc.reshape(list(enc.shape[:-2]) + [-1])
        if self.pca:
            self.pca["center"] = self.pca["center"].to(enc.device)
            self.pca["tensor"] = self.pca["tensor"].to(enc.device)
            out = torch.matmul(enc - self.pca["center"], self.pca["tensor"].T)
            return self.encoder(out)
        else:
            return self.encoder(enc)

    def decode(self, dec):
        if self.pca:
            decoded = torch.matmul(self.decoder(dec), self.pca["tensor"]) + self.pca["center"]
        else:
            decoded = self.decoder(dec)

        if self.datadim == 2:
            sqrt = int(np.sqrt(decoded.shape[-1]))
            decoded = decoded.reshape(list(dec.shape[:-1]) + [sqrt, -1])

        return decoded

    def save_model(self, filename):
        addr = f"{BASEDIR}/{filename}"

        dirpath = os.path.dirname(addr)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath)

        with open(addr, "wb") as handle:
            pickle.dump({"model": self}, handle, protocol=pickle.HIGHEST_PROTOCOL)


class FFVAE(nn.Module):
    def __init__(self, encodeSeq, decodeSeq, activation=nn.ReLU(), datadim=1, reg=0.1):
        super().__init__()
        self.activation = activation
        self.datadim = datadim
        self.reg = reg

        es = list(encodeSeq)
        self.latentdim = es[-1]
        es[-1] = 2*es[-1]

        self.reduced = self.latentdim
        self.input_dim = encodeSeq[0]

        self.encoder_net = FFNet(seq=es, activation=activation)
        self.decoder_net = FFNet(seq=decodeSeq, activation=activation)

    def encode(self, x, variance=False):
        if self.datadim == 2:
            batch = x.size(0)
            x = x.view(batch, -1)

        h = self.encoder_net(x)
        mu = h[..., :self.latentdim]
        logvar = h[..., self.latentdim:]

        if variance:
            return mu, logvar
        else:
            return mu

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, context=None):
        recon_flat = self.decoder_net(z)

        if self.datadim == 2:
            side = int(np.sqrt(self.input_dim))
            recon = recon_flat.view(-1, side, side)
            return recon

        return recon_flat

    def forward(self, x, variance=False):
        mu, logvar = self.encode(x, variance=True)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        if variance:
            return recon, mu, logvar
        else:
            return recon

    def loss_function(self, recon, x, mu, logvar):
        if self.datadim == 2:
            batch = x.size(0)
            x_flat = x.view(batch, -1)
        else:
            x_flat = x

        recon_flat = recon.view_as(x_flat)
        recon_loss = nn.MSELoss()(recon_flat, x_flat)

        sigma2 = torch.exp(logvar)
        kld_element = mu.pow(2) + sigma2 - 1 - logvar
        kld = 0.5 * torch.sum(kld_element)
        kld = (kld / x.size(0)) / recon_flat.shape[1]

        return (recon_loss, kld * self.reg)

    def save_model(self, filename):
        addr = os.path.join(BASEDIR, filename)
        os.makedirs(os.path.dirname(addr), exist_ok=True)

        with open(addr, "wb") as handle:
            pickle.dump({"model": self}, handle, protocol=pickle.HIGHEST_PROTOCOL)

Other_Modules = JC_Modules
