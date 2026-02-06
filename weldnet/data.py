"""
Data loading utilities with schema validation and standardized handling.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path

import numpy as np
import scipy.io
import h5py
import torch
import matplotlib.pyplot as plt
import ipywidgets as widgets
from scipy.io.matlab.miobase import MatReadWarning
from sklearn.decomposition import PCA
from matplotlib.colors import hsv_to_rgb
from omegaconf import DictConfig, OmegaConf


# ============================================================
# Path Configuration
# ============================================================

# Root directory for data files (project root's configs/data/)
_SRC_DIR = Path(__file__).parent
DATA_ROOT = _SRC_DIR.parent / "configs" / "data"


def resolve_data_path(path: str) -> Path:
    """
    Resolve a data file path to an absolute path.

    If path is absolute, return as-is.
    If path starts with '@/', resolve relative to DATA_ROOT.
    Otherwise, resolve relative to DATA_ROOT.

    Examples:
        resolve_data_path("@/transport/hatsscale.mat") -> DATA_ROOT/transport/hatsscale.mat
        resolve_data_path("transport/hatsscale.mat") -> DATA_ROOT/transport/hatsscale.mat
        resolve_data_path("/absolute/path/file.mat") -> /absolute/path/file.mat
    """
    path = str(path)

    if path.startswith("@/"):
        path = path[2:]  # Remove @/ prefix

    p = Path(path)
    if p.is_absolute():
        return p

    # Handle legacy relative paths like "../configs/data/..."
    if path.startswith("../configs/data/"):
        path = path.replace("../configs/data/", "")
    elif path.startswith("configs/data/"):
        path = path.replace("configs/data/", "")

    return DATA_ROOT / path


# ============================================================
# Config Schema
# ============================================================

@dataclass
class FileConfig:
    """Schema for file configuration."""
    filestr: str
    dataname: str = "alldata"
    name: str = ""

    def __post_init__(self):
        if not self.filestr:
            raise ValueError("filestr is required")
        if not self.name:
            # Auto-generate name from filename
            self.name = Path(self.filestr).stem


@dataclass
class DataSizeConfig:
    """Schema for data size/preprocessing configuration."""
    subset: Optional[int] = None  # Number of samples to use
    space: Optional[int] = None   # Spatial resolution
    time: Optional[int] = None    # Temporal resolution
    scaledown: bool = False       # Whether to normalize data
    spacedim: int = 1             # Spatial dimensions (1 or 2)


@dataclass
class DataConfig:
    """Schema for complete data configuration."""
    file: FileConfig
    datasize: DataSizeConfig = field(default_factory=DataSizeConfig)
    constants: dict = field(default_factory=dict)
    constraint: Optional[str] = None  # None, "cylinder", or "integral"

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        """Create DataConfig from a dictionary."""
        file_cfg = FileConfig(**d.get("file", {}))
        size_cfg = DataSizeConfig(**d.get("datasize", {}))
        constants = dict(d.get("constants", {}))
        constraint = d.get("constraint", None)
        return cls(file=file_cfg, datasize=size_cfg, constants=constants, constraint=constraint)

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "DataConfig":
        """Create DataConfig from OmegaConf DictConfig."""
        return cls.from_dict(OmegaConf.to_container(cfg, resolve=True))

    def validate(self) -> List[str]:
        """Validate config and return list of warnings/errors."""
        issues = []

        # Check file exists
        resolved_path = resolve_data_path(self.file.filestr)
        if not resolved_path.exists():
            issues.append(f"Data file not found: {resolved_path}")

        # Check spacedim
        if self.datasize.spacedim not in (1, 2):
            issues.append(f"spacedim must be 1 or 2, got {self.datasize.spacedim}")

        return issues


# ============================================================
# Data Loading
# ============================================================

def load_mat_file(path: Path, dataname: str = "alldata") -> tuple:
    """
    Load data from a .mat file.

    Returns:
        (data, params) tuple where params may be None
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MatReadWarning)
        matdata = scipy.io.loadmat(str(path))

    data = matdata[dataname]
    params = matdata.get("params", None)

    return data, params


def load_h5_file(path: Path, name: str = "") -> tuple:
    """
    Load data from an .h5 file.

    Returns:
        (data, params) tuple
    """
    with h5py.File(str(path), "r") as f:
        outputs = []
        for key in f.keys():
            outputs.append(np.asarray(f[key]["data"]).squeeze())
        data = np.stack(outputs, axis=0)

    # Special case handling for shallow water data
    if name == "shallowradsFULL":
        params = _compute_shallow_water_params(data)
    else:
        params = None

    return data, params


def _compute_shallow_water_params(data: np.ndarray) -> np.ndarray:
    """Compute radius parameters for shallow water data."""
    def radius_from_dist(outputs, unit_per_pixel=1.0):
        N = outputs.shape[0]
        cy = (N - 1) / 2.0
        cx = (N - 1) / 2.0
        ys, xs = np.nonzero(outputs - 1)
        d_pixels = np.hypot(ys - cy, xs - cx)
        r_pixels = d_pixels.max() + 0.5
        return r_pixels * unit_per_pixel

    units = 0.03875732421875
    rads = []
    for i in range(len(data)):
        rad = radius_from_dist(data[i, 0], unit_per_pixel=units)
        rads.append(rad)

    return np.asarray(rads)[:, None]


def standardize_params(params: Optional[np.ndarray], n_samples: int) -> np.ndarray:
    """
    Standardize params to always be a 2D array.

    If params is None, returns array of zeros with shape (n_samples, 1).
    If params is 1D, reshapes to (n_samples, 1).
    Reduces params to only varying columns.
    """
    if params is None:
        return np.zeros((n_samples, 1), dtype=np.float32)

    params = np.asarray(params, dtype=np.float32)

    if params.ndim == 1:
        params = params.reshape(-1, 1)

    # Reduce to only varying columns
    if params.shape[1] > 1:
        varying = np.ptp(params, axis=0) > 1e-4
        if varying.any():
            params = params[:, varying]

    return params


# ============================================================
# Utility functions for visualization methods
# ============================================================

def _pca_recon_error(data, k, ord=2):
    pca = PCA(n_components=k)
    principalComponents = pca.fit_transform(data)
    rdata = pca.inverse_transform(principalComponents)
    return np.mean(np.linalg.norm(rdata - data, axis=1, ord=ord) / np.linalg.norm(data, axis=1, ord=ord))

def _determine_params(paramarr):
    encoding_param = []
    P = paramarr.shape[1]
    if P == 1:
        return [0]
    for p in range(P):
        if np.abs(paramarr[0, p] - paramarr[1, p]) > 0:
            encoding_param.append(p)
    return encoding_param

def _twodim_colors(data):
    hue_min, hue_max = 0, 0.7
    sat_min, sat_max = 0.3, 1
    hue = np.interp(data[:, 0], (data[:, 0].min(), data[:, 0].max()), (hue_min, hue_max))
    sat = np.flip(np.interp(data[:, 1], (data[:, 1].min(), data[:, 1].max()), (sat_min, sat_max)))
    color_array = np.column_stack([hue, np.ones_like(hue), sat])
    return hsv_to_rgb(color_array)

def _link_3d_rotations(axes):
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
    return fig.canvas.mpl_connect("button_press_event", on_click)


# ============================================================
# DynamicData
# ============================================================

class DynamicData(torch.utils.data.Dataset):
    """
    Dataset class for dynamic PDE data.

    Supports loading from:
    - .mat files (scipy.io)
    - .h5 files (h5py)
    - Direct numpy arrays

    Args:
        config: Either a DataConfig, DictConfig, tuple (path, dataname), or tuple (data, params)
        seed: Random seed for shuffling
    """

    def __init__(self, config: Union[DataConfig, DictConfig, tuple], seed: int = 0, spacedim: int = 1):
        self.seed = seed
        self.scale = 1.0
        self.constants = {}
        self.constraint = None

        if isinstance(config, tuple):
            self._init_from_tuple(config)
        elif isinstance(config, DictConfig):
            self._init_from_omegaconf(config)
        elif isinstance(config, DataConfig):
            self._init_from_dataconfig(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        # Override spacedim if explicitly provided and not from DictConfig
        if spacedim != 1 and not isinstance(config, DictConfig):
            self.spacedim = spacedim

        # Standardize params
        self.params = standardize_params(self.params, len(self.data))

        # Ensure float32
        self.data = np.float32(self.data)
        self.params = np.float32(self.params)

        # Shuffle
        np.random.seed(seed)
        self._shuffle_inplace()

    def _init_from_tuple(self, config: tuple):
        """Initialize from tuple (path, dataname) or (data, params)."""
        if isinstance(config[0], str):
            # Use path as-is for direct tuple input (preserves notebook-relative paths)
            path = Path(config[0])
            self.data, self.params = load_mat_file(path, config[1])
            self.name = path.stem
        elif isinstance(config[0], np.ndarray):
            self.data = config[0]
            self.params = config[1]
            self.name = "direct_input"
        else:
            raise ValueError("Tuple must be (path, dataname) or (data, params)")

        self.spacedim = len(self.data.shape) - 2

    def _init_from_omegaconf(self, config: DictConfig):
        """Initialize from OmegaConf DictConfig (legacy support)."""
        dc = DataConfig.from_omegaconf(config)
        self._init_from_dataconfig(dc)

    def _init_from_dataconfig(self, config: DataConfig):
        """Initialize from DataConfig."""
        # Validate
        issues = config.validate()
        for issue in issues:
            warnings.warn(issue)

        # Load file
        path = resolve_data_path(config.file.filestr)
        suffix = path.suffix.lower()

        if suffix == ".mat":
            self.data, self.params = load_mat_file(path, config.file.dataname)
        elif suffix in (".h5", ".hdf5"):
            self.data, self.params = load_h5_file(path, config.file.name)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        self.name = config.file.name
        self.constants = config.constants

        # Set up constraint based on config
        self.constraint = self._create_constraint(config.constraint, config.constants)

        # Apply preprocessing
        ds = config.datasize
        if ds.subset:
            self._subset(ds.subset)
        if ds.space:
            self._downsample_space(ds.space)
        if ds.time:
            self._downsample_time(ds.time)
        if ds.scaledown:
            self.scaledown()

        self.spacedim = ds.spacedim

        # Reshape to correct spatial dimensions
        newshape = list(self.data.shape[:1 + self.spacedim]) + [-1]
        self.data = self.data.reshape(newshape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _create_constraint(self, constraint_type: Optional[str], constants: dict):
        """Create a constraint object based on type string."""
        if constraint_type is None:
            return None

        # Constraints are optional - only available in full codebase
        # For paper demo, constraints are not needed
        warnings.warn(f"Constraint '{constraint_type}' requested but constraints module not available in paper demo")
        return None

    def _subset(self, num: int):
        """Keep only first num samples."""
        self.data = self.data[:num]
        if self.params is not None:
            self.params = self.params[:num]

    def _downsample_space(self, target: int):
        """Downsample spatial dimensions to target resolution."""
        current = self.data.shape[2]
        factor = max(1, current // target)

        # Apply to all spatial dimensions
        ndim = len(self.data.shape)
        slices = [slice(None), slice(None)]  # batch, time
        for i in range(2, ndim):
            slices.append(slice(None, None, factor))

        self.data = self.data[tuple(slices)]

    def _downsample_time(self, target: int):
        """Downsample temporal dimension to target resolution."""
        current = self.data.shape[1]
        factor = max(1, current // target)
        self.data = self.data[:, ::factor]

    def scaledown(self):
        """Normalize data to [-1, 1] range."""
        self.scale = max(np.max(self.data), -np.min(self.data), 1e-8)
        self.data = self.data / self.scale

    def _shuffle_inplace(self):
        """Shuffle data and params together."""
        p = np.random.permutation(len(self.data))
        self.data = self.data[p]
        self.params = self.params[p]

    def get_from_params(self, params):
        """Find sample index closest to target parameter values (min-max normalized L1)."""
        params = np.asarray(params)
        minv = self.params.min(axis=0)
        maxv = self.params.max(axis=0)
        scale = maxv - minv
        scale[scale == 0] = 1.0
        norm_self = (self.params - minv) / scale
        norm_input = (params - minv) / scale
        a = np.linalg.norm(norm_self - norm_input, axis=1, ord=1)
        return int(np.argmin(a))

    def train_test_split(self, train_ratio: float = 0.9) -> tuple:
        """
        Split data into train and test sets.

        Returns:
            (train_data, test_data, train_params, test_params)
        """
        n_train = int(len(self.data) * train_ratio)
        return (
            self.data[:n_train],
            self.data[n_train:],
            self.params[:n_train],
            self.params[n_train:],
        )

    @property
    def shape(self):
        return self.data.shape

    # ========================================================
    # Visualization and analysis methods
    # ========================================================

    @property
    def is_2d_spatial(self):
        """Whether data has 2D spatial dimensions (rank 4)."""
        return len(self.data.shape) == 4

    def pca_error(self, k, ord=2):
        """Relative L2 PCA reconstruction error at latent dimension k."""
        flat = self.data.reshape(-1, self.data.shape[-1])
        return _pca_recon_error(flat, k, ord=ord)

    def determine_params(self):
        """Return indices of varying parameter columns."""
        return _determine_params(self.params)

    def reduce_params(self, tol=1e-4):
        """Return params array with only varying columns."""
        arr = np.asarray(self.params)
        if arr.ndim != 2:
            raise ValueError("params must be 2-D (N x p)")
        varying = np.ptp(arr, axis=0) > tol
        return arr[:, varying]

    def traintest_split_alltime(self, trainnum, t0=0, t1=-1, noise=0):
        if t1 == -1:
            t1 = self.data.shape[1] - 1
        noisedarray = self.data.copy()
        if noise > 0:
            noisedarray[:, 1:] += np.random.normal(scale=noise, size=noisedarray[:, 1:].shape)
        somedata = noisedarray[:, t0:t1]
        return np.split(somedata, [trainnum])

    def traintest_split_inout(self, trainnum, t0=0, t1=-1, noise=0):
        if t1 == -1:
            t1 = self.data.shape[1] - 1
        noisedarray = self.data.copy()
        if noise > 0:
            noisedarray[:, 1:] += np.random.normal(scale=noise, size=noisedarray[:, 1:].shape)
        somedata = np.column_stack([noisedarray[:, t0], noisedarray[:, t1]])
        return np.split(somedata, [trainnum])

    def collect_times(self):
        N, T = self.data.shape[:2]
        Ds = list(self.data.shape[2:])
        data = self.data.reshape([N*T] + Ds, order="F")
        if self.params is not None:
            params = np.column_stack([np.tile(self.params, (T, 1)), np.repeat(np.arange(0, T), N)])
            return data, params
        else:
            return data

    def plot_svd(self, title="", dpi=80, t=-1, maxnum=-1):
        if t >= 0:
            arr = self.data[:, t]
            arr = self.data.reshape(list(self.data.shape[:2]) + [-1])
        else:
            arr, _ = self.collect_times()
        arr = arr.reshape((arr.shape[0], -1))
        fig, ax = plt.subplots(figsize=(7, 3), dpi=dpi)
        if maxnum > 0:
            arr = arr[:maxnum]
        u, s, vh = np.linalg.svd(arr, full_matrices=False)
        ax.plot(s, color="blue")
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value Magnitude")
        ax.grid(True, which="major", linestyle="--", linewidth=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        fig.tight_layout()

    def plot_pca3_pt(self, param=0, s=0):
        plt.rcParams.update({'font.size': 14})
        pca = PCA(n_components=s+3)
        data, params = self.collect_times()
        data = data.reshape((data.shape[0], -1))
        principalComponents = pca.fit_transform(data)
        x = principalComponents[:, s+0]
        y = principalComponents[:, s+1]
        z = principalComponents[:, s+2]
        rdata = pca.inverse_transform(principalComponents)
        rerror = np.mean(np.linalg.norm(rdata - data, axis=1, ord=2) / np.linalg.norm(data, axis=1, ord=2))
        print("Reconstruction error", rerror)
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(projection='3d')
        ptcolors = _twodim_colors(params[:, (param, -1)])
        sc = ax.scatter(x, y, z, c=ptcolors, s=2)
        ax.set_xlabel(f'Component {s+1}')
        ax.set_ylabel(f'Component {s+2}')
        ax.set_zlabel(f'Component {s+3}')
        ax.set_box_aspect((1.3, 1.3, 1))
        ax.view_init(elev=30, azim=45)
        ax.set_title(f"Parameter {param} and Time")

    def plot_pca3(self, param=-1, s=0, t=-1):
        if param == -1:
            param = self.determine_params()[0]
        plt.rcParams.update({'font.size': 14})
        pca = PCA(n_components=s+3)
        if t >= 0:
            data = self.data[:, t]
            params = self.params
        else:
            data, params = self.collect_times()
        data = data.reshape((data.shape[0], -1))
        principalComponents = pca.fit_transform(data)
        x = principalComponents[:, s+0]
        y = principalComponents[:, s+1]
        z = principalComponents[:, s+2]
        rdata = pca.inverse_transform(principalComponents)
        rerror = np.mean(np.linalg.norm(rdata - data, axis=1, ord=2) / np.linalg.norm(data, axis=1, ord=2))
        print("Reconstruction error", rerror)
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122, projection='3d')
        sc = ax.scatter(x, y, z, c=params[:, param], s=2)
        plt.colorbar(sc, ax=ax, location="right", pad=0)
        ax.set_xlabel(f'Component {s+1}')
        ax.set_ylabel(f'Component {s+2}')
        ax.set_zlabel(f'Component {s+3}')
        ax.set_box_aspect((1.3, 1.3, 1))
        ax.view_init(elev=30, azim=45)
        ax.set_title(f"Parameter {param}")
        if t == -1:
            sc1 = ax1.scatter(x, y, z, c=params[:, -1], s=2, cmap="copper")
            plt.colorbar(sc1, ax=ax1, location="right", pad=0)
            ax1.set_xlabel(f'Component {s+1}')
            ax1.set_ylabel(f'Component {s+2}')
            ax1.set_zlabel(f'Component {s+3}')
            ax1.set_box_aspect((1.3, 1.3, 1))
            ax1.view_init(elev=30, azim=45)
            ax1.set_title("Time")
        return (fig, ax)

    def plot_pca2_pt(self, param=0, s=0):
        plt.rcParams.update({'font.size': 14})
        pca = PCA(n_components=s+2)
        data, params = self.collect_times()
        data = data.reshape((data.shape[0], -1))
        principalComponents = pca.fit_transform(data)
        x = principalComponents[:, s+0]
        y = principalComponents[:, s+1]
        rdata = pca.inverse_transform(principalComponents)
        rerror = np.mean(np.linalg.norm(rdata - data, axis=1, ord=2) / np.linalg.norm(data, axis=1, ord=2))
        print("Reconstruction error", rerror)
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot()
        ptcolors = _twodim_colors(params[:, (param, -1)])
        sc = ax.scatter(x, y, c=ptcolors, s=2)
        ax.set_xlabel(f'Component {s+1}')
        ax.set_ylabel(f'Component {s+2}')
        ax.set_title(f"Parameter {param} and Time")

    def plot_pca2(self, param=0, s=0, t=-1):
        plt.rcParams.update({'font.size': 14})
        pca = PCA(n_components=s+2)
        if t >= 0:
            data = self.data[:, t]
            params = self.params
        else:
            data, params = self.collect_times()
        data = data.reshape((data.shape[0], -1))
        principalComponents = pca.fit_transform(data)
        x = principalComponents[:, s+0]
        y = principalComponents[:, s+1]
        rdata = pca.inverse_transform(principalComponents)
        rerror = np.mean(np.linalg.norm(rdata - data, axis=1, ord=2) / np.linalg.norm(data, axis=1, ord=2))
        print("Reconstruction error", rerror)
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(121)
        ax1 = fig.add_subplot(122)
        sc = ax.scatter(x, y, c=params[:, param], s=1)
        plt.colorbar(sc, ax=ax, location="right", pad=0)
        ax.set_xlabel(f'Component {s+1}')
        ax.set_ylabel(f'Component {s+2}')
        ax.set_title(f"Parameter {param}")
        if t < 0:
            sc = ax1.scatter(x, y, c=params[:, -1], s=1, cmap="copper")
            plt.colorbar(sc, ax=ax1, location="right", pad=0)
            ax1.set_xlabel(f'Component {s+1}')
            ax1.set_ylabel(f'Component {s+2}')
            ax1.set_title("Time")
        return fig

    def plot_data_3d(self, noise=0, mode="normal", yscalefixed=False, surf=False):
        assert len(self.data.shape) >= 4, "plot_data_3d requires data with rank >= 4"
        noisedarray = np.copy(self.data)
        noisedarray[:, 1:] += np.random.normal(scale=noise, size=noisedarray[:, 1:].shape)
        cmin, cmax = np.min(noisedarray), np.max(noisedarray)
        if surf:
            fig = plt.figure(figsize=(9, 3))
            ax0 = fig.add_subplot(1, 2, 1, projection='3d')
            ax1 = fig.add_subplot(1, 2, 2, projection='3d')
            axes = [ax0, ax1]
            _link_3d_rotations(axes)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(7, 3))

        def _plot_surfaces(j, t0, t1):
            H, W = noisedarray.shape[2], noisedarray.shape[3]
            X = np.arange(W)
            Y = np.arange(H)
            X, Y = np.meshgrid(X, Y)
            axes[0].plot_surface(X, Y, noisedarray[j, t0], cmap='jet', vmin=cmin, vmax=cmax, linewidth=0, antialiased=False)
            axes[0].set_title(f"t={t0}")
            axes[1].plot_surface(X, Y, noisedarray[j, t1], cmap='jet', vmin=cmin, vmax=cmax, linewidth=0, antialiased=False)
            axes[1].set_title(f"t={t1}")
            if yscalefixed:
                axes[0].set_zlim(cmin, cmax)
                axes[1].set_zlim(cmin, cmax)

        if mode == "normal":
            @widgets.interact(j=(0, noisedarray.shape[0] - 1), t0=(0, noisedarray.shape[1] - 1), t1=(0, noisedarray.shape[1] - 1))
            def plot_io(j=0, t0=0, t1=1):
                axes[0].cla(); axes[1].cla()
                if surf:
                    _plot_surfaces(j, t0, t1)
                else:
                    axes[0].imshow(noisedarray[j, t0], vmin=cmin, vmax=cmax, cmap="jet", origin="lower"); axes[0].set_title(f"t={t0}")
                    axes[1].imshow(noisedarray[j, t1], vmin=cmin, vmax=cmax, cmap="jet", origin="lower"); axes[1].set_title(f"t={t1}")
                fig.suptitle(f"Params: {self.params[j]}" if self.params is not None else f"Data point {j}")
                fig.tight_layout()
        elif mode == "params":
            sliders = {"t0": widgets.IntSlider(value=0, min=0, max=noisedarray.shape[1] - 1),
                        "t1": widgets.IntSlider(value=1, min=0, max=noisedarray.shape[1] - 1)}
            for i in range(self.params.shape[1]):
                minval, maxval = float(np.min(self.params[:, i])), float(np.max(self.params[:, i]))
                sliders[f"p{i}"] = widgets.FloatSlider(value=minval, min=minval, max=maxval, step=0.01)
            @widgets.interact(t0=sliders['t0'], t1=sliders['t1'], **{k: sliders[k] for k in sliders if k.startswith('p')})
            def update(t0=0, t1=1, **args):
                axes[0].cla(); axes[1].cla()
                j = self.get_from_params(np.array(list(args.values())))
                if surf:
                    _plot_surfaces(j, t0, t1)
                else:
                    axes[0].imshow(noisedarray[j, t0], vmin=cmin, vmax=cmax, cmap="jet", origin="lower"); axes[0].set_title(f"t={t0}")
                    axes[1].imshow(noisedarray[j, t1], vmin=cmin, vmax=cmax, cmap="jet", origin="lower"); axes[1].set_title(f"t={t1}")
                fig.suptitle(f"Params: {self.params[j]}")
                fig.tight_layout()

    def plot_data_2d(self, noise=0, mode="normal", topdown=False, yscalefixed=False):
        assert len(self.data.shape) == 3, "plot_data_2d requires data with rank == 3"
        noisedarray = np.copy(self.data)
        noisedarray[:, 1:] += np.random.normal(scale=noise, size=noisedarray[:, 1:].shape)
        cmin, cmax = np.min(noisedarray), np.max(noisedarray)
        fig, ax = plt.subplots(figsize=(4, 3))

        if mode == "normal":
            @widgets.interact(j=(0, noisedarray.shape[0] - 1), t0=(0, noisedarray.shape[1] - 1), t1=(0, noisedarray.shape[1] - 1))
            def plot_io(j=0, t0=0, t1=1):
                ax.clear()
                if topdown:
                    ax.imshow(noisedarray[j], aspect='auto', vmin=cmin, vmax=cmax, cmap='jet', origin="lower")
                    ax.set_title(f"Data point {j} over all time vs. features")
                else:
                    ax.plot(noisedarray[j, t0, :], 'r', label=f"t={t0}")
                    ax.plot(noisedarray[j, t1, :], 'b', label=f"t={t1}")
                    ax.legend()
                    if yscalefixed:
                        ax.set_ylim(cmin, cmax)
                fig.suptitle(f"Params: {self.params[j]}" if self.params is not None else f"Data point {j}")
                fig.tight_layout()
        elif mode == "params":
            sliders = {"t0": widgets.IntSlider(value=0, min=0, max=noisedarray.shape[1] - 1),
                        "t1": widgets.IntSlider(value=1, min=0, max=noisedarray.shape[1] - 1)}
            for i in range(self.params.shape[1]):
                minval, maxval = float(np.min(self.params[:, i])), float(np.max(self.params[:, i]))
                sliders[f"p{i}"] = widgets.FloatSlider(value=minval, min=minval, max=maxval, step=0.01)
            @widgets.interact(t0=sliders['t0'], t1=sliders['t1'], **{k: sliders[k] for k in sliders if k.startswith('p')})
            def update(t0=0, t1=1, **args):
                ax.clear()
                j = self.get_from_params(np.array(list(args.values())))
                if topdown:
                    ax.imshow(noisedarray[j], aspect='auto', vmin=cmin, vmax=cmax, cmap='jet', origin="lower")
                else:
                    ax.plot(noisedarray[j, t0, :], 'r', label=f"t={t0}")
                    ax.plot(noisedarray[j, t1, :], 'b', label=f"t={t1}")
                    ax.legend()
                    if yscalefixed:
                        ax.set_ylim(cmin, cmax)
                fig.suptitle(f"Params: {self.params[j]}")
                fig.tight_layout()

    def plot_data(self, *args, **kwargs):
        """Dispatcher: calls plot_data_3d or plot_data_2d based on rank."""
        rank = len(self.data.shape)
        if rank >= 4:
            return self.plot_data_3d(*args, **kwargs)
        elif rank == 3:
            return self.plot_data_2d(*args, **kwargs)
        else:
            raise ValueError("Data must be 3D or 4D+ for plotting")

    def plot_data_static(self, samples=None, param_label="a", n_times=5, n_samples=3,
                         tmax=1.0, xmax=1.0, ymax=1.0):
        """Static publication-quality plots."""
        if self.is_2d_spatial:
            from .utils import plot_data_2d as _static_plot_2d
            return _static_plot_2d(self, samples, param_label, n_times, n_samples, tmax, xmax, ymax)
        # 1D case: line curves
        _, T, X = self.data.shape
        if samples is None:
            params = self.params[:, 0] if len(self.params.shape) > 1 else self.params
            sorted_idx = np.argsort(params)
            samples = [sorted_idx[int(i)] for i in np.linspace(0, len(sorted_idx) - 1, n_samples)]
        time_indices = np.linspace(0, T - 1, n_times, dtype=int)
        time_phys = [tmax * t / (T - 1) for t in time_indices]
        x = np.linspace(0, xmax, X)
        fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 3))
        if len(samples) == 1:
            axes = [axes]
        for ax, s in zip(axes, samples):
            for t_idx, t_phys in zip(time_indices, time_phys):
                ax.plot(x, self.data[s, t_idx], alpha=0.8, label=f'$t$={t_phys:.2f}')
            param_val = self.params[s, 0] if len(self.params.shape) > 1 else self.params[s]
            ax.set_title(f"${param_label}$ = {param_val:.3f}")
            ax.set_xlabel("$x$")
            ax.set_ylabel("$u$")
        axes[0].legend(fontsize=7)
        plt.tight_layout()
        return fig

    def get_pcaerrors(self, k=10):
        data = self.data
        pca = PCA(n_components=k)
        pca = pca.fit(data.reshape(-1, data.shape[-1]))
        pcaerrs = []
        for t in range(data.shape[1]):
            dataslice = data[:, t]
            components = pca.transform(dataslice)
            rdata = pca.inverse_transform(components)
            pcaerrs.append(np.mean(np.linalg.norm(rdata - dataslice) / np.linalg.norm(dataslice)))
        fig, ax = plt.subplots()
        times = np.arange(data.shape[1])
        ax.set_xlabel("Time")
        ax.plot(times, pcaerrs, marker='o', label=f"PCA{k}")
        ax.set_ylabel("RelL2 Reconstruction Error")
        ax.legend()
        fig.tight_layout()

    def __repr__(self):
        return (
            f"DynamicData(name={self.name!r}, shape={self.shape}, "
            f"params_shape={self.params.shape}, spacedim={self.spacedim})"
        )
