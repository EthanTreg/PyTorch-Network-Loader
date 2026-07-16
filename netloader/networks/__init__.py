"""
Collects all architectures
"""
from warnings import warn

from netloader.architectures import (
    BaseArchitecture,
    UtilityMixin,
    Autoencoder,
    Decoder,
    Encoder,
    load_arch,
)

BaseNetwork = BaseArchitecture
load_net = load_arch


warn(
    'Importing architectures from netloader.networks is deprecated, please use '
    'netloader.architectures',
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    'UtilityMixin',
    'BaseNetwork',
    'Autoencoder',
    'Decoder',
    'Encoder',
    'load_net',
]

_optional_imports: dict[str, set[str]] = {'flow': {'NormFlow', 'NormFlowEncoder'}}

try:
    from netloader.architectures.flows import NormFlow, NormFlowEncoder
    __all__.extend(['NormFlow', 'NormFlowEncoder'])
except (ModuleNotFoundError, ImportError):
    pass


def __getattr__(name):
    if name not in __all__ and name in _optional_imports['flow']:
        raise ImportError(f'Cannot import {name}, normalising flow architectures require the '
                          f'package Zuko')
    raise AttributeError(f"module {__name__} has no attribute {name}")
