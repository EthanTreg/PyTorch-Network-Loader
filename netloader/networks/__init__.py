"""
Collects all networks
"""
from netloader.networks.base import BaseNetwork
from netloader.networks.encoder_decoder import Autoencoder, Decoder, Encoder

try:
    from netloader.networks.flows import NormFlow, NormFlowEncoder, norm_flow
except ImportError:
    pass
