"""
Package information, creates the logger and adds netloader classes to PyTorch safe globals
"""
import sys
import inspect
import logging


__version__ = '3.5.7'
__author__ = 'Ethan Tregidga'
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

try:
    import torch

    import netloader.networks as nets
    from netloader.network import Network
    from netloader import transforms, loss_funcs


    # Adds PyTorch Network Loader classes to list of safe PyTorch classes when loading saved
    # networks
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
except ModuleNotFoundError:
    pass
except ImportError:
    pass
