"""
Package information, creates the logger and adds netloader classes to PyTorch safe globals
"""
import logging


__version__ = '3.8.0'
__author__ = 'Ethan Tregidga'
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

try:
    import torch

    from netloader.network import Network
    from netloader.utils.utils import safe_globals
    from netloader import networks, transforms, loss_funcs, models


    # Adds PyTorch Network Loader classes to list of safe PyTorch classes when loading saved
    # networks
    safe_globals(__name__, [networks, transforms, loss_funcs, models])
    torch.serialization.add_safe_globals([Network])
except ModuleNotFoundError:
    pass
except ImportError:
    pass
