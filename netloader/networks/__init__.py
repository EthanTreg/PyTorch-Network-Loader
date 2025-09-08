"""
Collects all networks
"""
from netloader.networks.utils import UtilityMixin
from netloader.networks.base import BaseNetwork, load_net
from netloader.networks.encoder_decoder import Autoencoder, Decoder, Encoder

__all__ = [
    'BaseNetwork',
    'UtilityMixin',
    'Autoencoder',
    'Decoder',
    'Encoder',
    'load_net',
]

try:
    from netloader.networks.flows import NormFlow, NormFlowEncoder
    __all__.extend(['NormFlow', 'NormFlowEncoder'])
except ImportError:
    pass
