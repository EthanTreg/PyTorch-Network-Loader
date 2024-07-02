"""
Misc functions used elsewhere
"""
from typing import Any
from types import ModuleType

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray


def get_device() -> tuple[dict[str, Any], torch.device]:
    """
    Gets the device for PyTorch to use

    Returns
    -------
    tuple[dict, device]
        Arguments for the PyTorch DataLoader to use when loading data into memory and PyTorch device
    """
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kwargs: dict[str, Any] = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
    return kwargs, device


def label_change(
        data: ndarray | Tensor,
        in_label: ndarray | Tensor,
        one_hot: bool = False,
        out_label: ndarray | Tensor | None = None) -> ndarray | Tensor:
    """
    Converts an array or tensor of class values to an array or tensor of class indices

    Parameters
    ----------
    data : N ndarray | Tensor
        Classes of size N
    in_label : C ndarray | Tensor
        Unique class values of size C found in data
    one_hot : bool, default = False
        If the returned tensor should be 1D array of class indices or 2D one hot tensor if out_label
        is None or is an int
    out_label : C ndarray | Tensor, default = None
        Unique class values of size C to transform data into, if None, then values will be indexes

    Returns
    -------
    N | NxC ndarray | Tensor
        ndarray or Tensor of class indices, or if one_hot is True, one hot tensor
    """
    data_one_hot: ndarray | Tensor
    out_data: ndarray | Tensor
    module: ModuleType

    if isinstance(data, Tensor):
        module = torch
    elif isinstance(data, ndarray):
        module = np
    else:
        raise TypeError(f'Data type {type(data)} not supported')

    if out_label is None:
        out_label = module.arange(len(in_label))

    if isinstance(out_label, Tensor):
        out_label = out_label.to(get_device()[1])

    out_data = out_label[module.searchsorted(in_label, data)]

    if one_hot:
        data_one_hot = module.zeros((len(data), len(in_label)))
        data_one_hot[module.arange(len(data)), out_data] = 1
        out_data = data_one_hot

    if isinstance(out_data, Tensor):
        out_data = out_data.to(get_device()[1])

    return out_data


def progress_bar(i: int, total: int, text: str = '') -> None:
    """
    Terminal progress bar

    Parameters
    ----------
    i : int
        Current progress
    total : int
        Completion number
    text : str, default = ''
        Optional text to place at the end of the progress bar
    """
    filled: int
    length: int = 50
    percent: float
    bar_fill: str
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar_fill = '█' * filled + '-' * (length - filled)
    print(f'\rProgress: |{bar_fill}| {int(percent)}%\t{text}\t', end='')

    if i == total:
        print()


def save_name(num: int, states_dir: str, name: str) -> str:
    """
    Standardises the network save file naming

    Parameters
    ----------
    num : int
        File number
    states_dir : str
        Directory of network saves
    name : str
        Name of the network

    Returns
    -------
    str
        Path to the network save file
    """
    return f'{states_dir}{name}_{num}.pth'
