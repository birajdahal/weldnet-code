"""
WeldNet: Windowed Encoders for Learning Dynamics
=================================================

This package implements WeldNet for data-driven model reduction
as described in arXiv:2512.11090.

Core Classes:
    - WeldNet: Main windowed autoencoder architecture
    - WindowTrajectory: Base class for window-based trajectory operations
    - WeldHelper: Helper class for training and evaluation

Autoencoders:
    - PCAAutoencoder: PCA-based dimensionality reduction
    - FFAutoencoder: Feed-forward autoencoder
    - FFVAE: Variational autoencoder
    - TCAutoencoderConv: Trigonometric basis convolutional autoencoder

Comparison Models:
    - LDNet: Grid-based latent dynamics network
    - TimeInputModel: Direct time-to-output mapping
    - HighDimProp: High-dimensional propagator
    - LDON: Latent Deep Operator Network

Networks:
    - FFNet: Feed-forward network
    - DeepONet: Deep operator network (Latent DON)

Data:
    - DynamicData: Dynamic data loader with visualization
    - DataConfig: Data configuration utilities
"""

# Core WeldNet classes
from .weld import (
    WindowTrajectory,
    WeldNet,
    WeldHelper,
)

# Base networks, autoencoders, and utilities
from .base import (
    FFNet,
    DeepONet,
    get_activation,
    get_actvn,
    ConvEncoder,
    ConvDecoder,
    BaseHelper,
    Comparisons,
    train_pca_shared,
    compute_relative_errors,
    JC_Modules,
    PCAAutoencoder,
    FFAutoencoder,
    FFVAE,
    Other_Modules,
)

# Other models: TimeInput, HighDimProp, LDNet, LDON
from .othermodels import (
    TimeInputModel,
    TimeInputHelper,
    HighDimProp,
    HighDimPropHelper,
    LDNet,
    LDHelper,
    LDON,
    LDONHelper,
)

# Data loading and utilities
from .data import (
    DynamicData,
    DataConfig,
    resolve_data_path,
)

from .utils import (
    num_params,
    # Plotting functions
    plot_prediction,
    plot_compare,
    plot_prediction_1d,
    plot_prediction_2d,
    plot_compare_1d,
    plot_compare_2d,
)

# Configuration utilities
from .config import (
    load_config,
    create_object,
    create_dataset,
)

__version__ = "0.1.0"
__all__ = [
    # Core WeldNet
    "WindowTrajectory",
    "WeldNet",
    "WeldHelper",
    # Autoencoders
    "PCAAutoencoder",
    "FFAutoencoder",
    "FFVAE",
    "Other_Modules",
    # Networks
    "FFNet",
    "DeepONet",
    "get_activation",
    "get_actvn",
    "ConvEncoder",
    "ConvDecoder",
    "BaseHelper",
    "Comparisons",
    "JC_Modules",
    # Latent Dynamics
    "LDNet",
    "LDHelper",
    # Baselines
    "TimeInputModel",
    "TimeInputHelper",
    "HighDimProp",
    "HighDimPropHelper",
    # LDON
    "LDON",
    "LDONHelper",
    # Data
    "DynamicData",
    "DataConfig",
    "resolve_data_path",
    # Utilities
    "num_params",
    "train_pca_shared",
    "compute_relative_errors",
    # Plotting
    "plot_prediction",
    "plot_compare",
    "plot_prediction_1d",
    "plot_prediction_2d",
    "plot_compare_1d",
    "plot_compare_2d",
    # Config
    "load_config",
    "create_object",
    "create_dataset",
]
