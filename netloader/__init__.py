"""
Package information and creates the logger
"""
import sys
import inspect
import logging

import torch

import netloader.networks as nets
from netloader import transforms
from netloader import loss_funcs
from netloader.network import Network


__version__ = '3.4.0'
__author__ = 'Ethan Tregidga'
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

# Adds PyTorch Network Loader classes to list of safe PyTorch classes when loading saved networks
torch.serialization.add_safe_globals([
    net[1] for net in inspect.getmembers(sys.modules[nets.__name__]) if isinstance(net[1], type)
])
torch.serialization.add_safe_globals([
    transform[1] for transform in inspect.getmembers(sys.modules[transforms.__name__])
    if isinstance(transform[1], type)
])
torch.serialization.add_safe_globals([
    loss_func[1] for loss_func in inspect.getmembers(sys.modules[loss_funcs.__name__])
    if isinstance(loss_func[1], type)
])
torch.serialization.add_safe_globals([Network])
