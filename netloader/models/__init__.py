"""
Collects all classes in the models module
"""
from netloader.models.misc import MultiNetwork
from netloader.models.convnext import (
    ConvNeXt,
    ConvNeXtTiny,
    ConvNeXtBase,
    ConvNeXtSmall,
    ConvNeXtLarge,
    ConvNeXtXLarge,
)

__all__ = [
    'ConvNeXt',
    'ConvNeXtTiny',
    'ConvNeXtBase',
    'MultiNetwork',
    'ConvNeXtSmall',
    'ConvNeXtLarge',
    'ConvNeXtXLarge',
]
