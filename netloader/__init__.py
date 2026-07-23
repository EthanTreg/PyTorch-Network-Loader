"""
Package information, creates the logger and adds netloader classes to PyTorch safe globals
"""
import sys
import logging
import warnings


__version__ = '3.11.3'
__author__ = 'Ethan Tregidga'
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)
warnings.filterwarnings('once', category=DeprecationWarning, module=r'^netloader(\.|$)')
warnings.filterwarnings(
    'once',
    category=PendingDeprecationWarning,
    module=r'^netloader(\.|$)',
)


if sys.version_info < (3, 12):
    warnings.warn(
        f'Python version ({sys.version_info.major}.{sys.version_info.minor}) is '
        f'deprecated, please upgrade to Python 3.12 or higher',
        DeprecationWarning,
    )

try:
    import torch

    from netloader.utils import safe_globals
    from netloader import (
        layers,
        models,
        architectures,
        utils,
        data,
        loss_funcs,
        network,
        schedulers,
        transforms,
    )


    # Adds PyTorch Network Loader classes to list of safe PyTorch classes when loading saved
    # architectures
    safe_globals(
        __name__,
        [models, loss_funcs, schedulers, transforms],
        use_all=True,
    )
    safe_globals(
        __name__,
        [architectures],
        use_all=True,
        legacy_path_replace=('architectures', 'networks'),
    )
    torch.serialization.add_safe_globals([
        network.BaseNetwork,
        network.Network,
        network.CompatibleNetwork,
    ])
    torch.serialization.add_safe_globals([])

    __all__ = [
        'utils',
        'layers',
        'models',
        'architectures',
        'data',
        'network',
        'loss_funcs',
        'schedulers',
        'transforms',
    ]
except (ModuleNotFoundError, ImportError):
    pass
