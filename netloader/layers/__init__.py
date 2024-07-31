"""
Collects all functions in the layers module
"""
from netloader.layers.linear import Activation, Linear, OrderedBottleneck, Sample, Upsample
from netloader.layers.pooling import AdaptivePool, Pool, PoolDownscale
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
    Reshape,
    Scale,
    Shortcut,
    Skip,
    Unpack,
)

try:
    from netloader.layers.flows import SplineFlow
except ImportError:
    pass
