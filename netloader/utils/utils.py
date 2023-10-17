"""
Misc functions used elsewhere
"""
import torch


def get_device() -> tuple[dict, torch.device]:
    """
    Gets the device for PyTorch to use

    Returns
    -------
    tuple[dictionary, device]
        Arguments for the PyTorch DataLoader to use when loading data into memory and PyTorch device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    return kwargs, device
