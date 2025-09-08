"""
Collects all classes in the layers module
"""
from netloader.layers.linear import Activation, Linear, OrderedBottleneck, Sample, Upsample
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
    Checkpoint,
    Concatenate,
    DropPath,
    Index,
    LayerNorm,
    Pack,
    Reshape,
    Scale,
    Shortcut,
    Skip,
    Unpack,
)


__all__ = [
    'Activation',
    'AdaptivePool',
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
    'DropPath',
    'Index',
    'LayerNorm',
    'Pack',
    'Linear',
    'OrderedBottleneck',
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
]


try:
    from netloader.layers.flows import SplineFlow
    __all__.append('SplineFlow')
except ImportError:
    pass
