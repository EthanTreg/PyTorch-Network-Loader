"""
Collects all architectures
"""
from typing import Any
from warnings import warn

from netloader.architectures.utils import UtilityMixin
from netloader.architectures.base import BaseArchitecture, load_arch
from netloader.architectures.encoder_decoder import Autoencoder, Decoder, Encoder


__all__ = [
    'BaseArchitecture',
    'UtilityMixin',
    'Autoencoder',
    'Decoder',
    'Encoder',
    'load_arch',
]

_optional_imports: dict[str, set[str]] = {'flow': {'NormFlow', 'NormFlowEncoder'}}

try:
    from netloader.architectures.flows import NormFlow, NormFlowEncoder
    __all__.extend(_optional_imports['flow'])
except (ModuleNotFoundError, ImportError) as e:
    pass


def __getattr__(name: str) -> Any:
    if name == 'BaseNetwork':
        warn(
            'BaseNetwork is deprecated, please use BaseArchitecture instead',
            DeprecationWarning,
            stacklevel=2,
        )
        return BaseArchitecture
    if name == 'load_net':
        warn(
            'load_net is deprecated, please use load_arch instead',
            DeprecationWarning,
            stacklevel=2,
        )
        return load_arch
    if name not in __all__ and name in _optional_imports['flow']:
        raise ImportError(f'Cannot import {name}, normalising flow architectures require the '
                          f'package Zuko')
    raise AttributeError(f"module {__name__} has no attribute {name}")
