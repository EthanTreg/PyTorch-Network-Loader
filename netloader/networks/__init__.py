"""
Collects all networks
"""
from netloader.networks.base import BaseNetwork, load_net
from netloader.networks.encoder_decoder import Autoencoder, Decoder, Encoder

try:
    from netloader.networks.flows import NormFlow, NormFlowEncoder
except ImportError:
    pass
