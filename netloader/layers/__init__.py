"""
Collects all classes in the layers module
"""
from netloader.layers.linear import Activation, Linear, OrderedBottleneck, Sample, Upsample
from netloader.layers.base import BaseLayer, BaseSingleLayer, BaseMultiLayer
from netloader.layers.pooling import AdaptivePool, Pool, PoolDownscale
from netloader.layers.blocks import ConvNeXtBlock
from netloader.layers.recurrent import Recurrent
from netloader.layers.convolutional import (
    Conv,
    ConvDepth,
    ConvDepthDownscale,
    ConvDownscale,
    ConvTranspose,
    ConvTransposeUpscale,
    ConvUpscale,
    PixelShuffle,
)
from netloader.layers.misc import (
    BatchNorm,
    Checkpoint,
    Concatenate,
    Dropout,
    DropPath,
    Index,
    LayerNorm,
    Pack,
    Pad,
    Reshape,
    Scale,
    Shortcut,
    Skip,
    Unpack,
)
from netloader.layers import utils


__all__ = [
    'Activation',
    'AdaptivePool',
    'BaseLayer',
    'BaseMultiLayer',
    'BaseSingleLayer',
    'BatchNorm',
    'Checkpoint',
    'Concatenate',
    'Conv',
    'ConvDepth',
    'ConvDepthDownscale',
    'ConvDownscale',
    'ConvNeXtBlock',
    'ConvTranspose',
    'ConvTransposeUpscale',
    'ConvUpscale',
    'Dropout',
    'DropPath',
    'Index',
    'LayerNorm',
    'Linear',
    'OrderedBottleneck',
    'Pack',
    'Pad',
    'PixelShuffle',
    'Pool',
    'PoolDownscale',
    'Recurrent',
    'Reshape',
    'Sample',
    'Scale',
    'Shortcut',
    'Skip',
    'Unpack',
    'Upsample',
    'utils',
]


try:
    from netloader.layers.flows import SplineFlow
    __all__.append('SplineFlow')
except ImportError:
    pass
