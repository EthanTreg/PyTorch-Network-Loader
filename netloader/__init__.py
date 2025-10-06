"""
Package information, creates the logger and adds netloader classes to PyTorch safe globals
"""
import logging
import warnings


__version__ = '3.9.1'
__author__ = 'Ethan Tregidga'
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)
warnings.filterwarnings('once', category=DeprecationWarning, module=r'^netloader(\.|$)')
warnings.filterwarnings(
    'once',
    category=PendingDeprecationWarning,
    module=r'^netloader(\.|$)',
)

try:
    import torch

    from netloader.network import Network
    from netloader.utils import safe_globals
    from netloader import networks, transforms, loss_funcs, models


    # Adds PyTorch Network Loader classes to list of safe PyTorch classes when loading saved
    # networks
    safe_globals(__name__, [networks, transforms, loss_funcs, models])
    torch.serialization.add_safe_globals([Network])
except ModuleNotFoundError:
    pass
except ImportError:
    pass
