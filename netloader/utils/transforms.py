"""
Deprecated module for backwards compatibility
"""
from warnings import warn

# pylint: disable=unused-import
from netloader.transforms import (
    BaseTransform,
    Log,
    MinClamp,
    MultiTransform,
    Normalise,
    NumpyTensor,
)


warn(
    'Importing transforms from utils is deprecated, please use netloader.transforms',
    DeprecationWarning,
    stacklevel=2,
)
