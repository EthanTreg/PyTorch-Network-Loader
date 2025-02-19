"""
Misc functions used elsewhere
"""
import os
import logging as log
from types import ModuleType
from typing import Any, TypeVar

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray

ArrayLike = TypeVar('ArrayLike', ndarray, Tensor)


def check_params(name: str, supported_params: list[str] | ndarray, in_params: ndarray) -> None:
    """
    Checks if provided parameters are supported by the function

    Parameters
    ----------
    name : str
        Name of the function
    supported_params : list[str] | ndarray
        Parameters supported by the function
    in_params : ndarray
        Input parameters
    """
    bad_params: ndarray = in_params[~np.isin(in_params, supported_params)]

    if len(bad_params):
        log.getLogger(__name__).warning(f'Unknown parameters for {name}: {bad_params}')


def deep_merge(base: dict, new: dict) -> dict:
    """
    Performs a deep merge of two dictionaries, equivalent to recursive base | new

    Parameters
    ----------
    base : dict
        Base dictionary
    new : dict
        Dictionary to deep merge into base

    Returns
    -------
    dict
        Deep merged dictionary
    """
    merged: dict = base.copy()
    key: Any
    value: Any

    for key, value in new.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_device() -> tuple[dict[str, Any], torch.device]:
    """
    Gets the device for PyTorch to use

    Returns
    -------
    tuple[dict[str, Any], device]
        Arguments for the PyTorch DataLoader to use when loading data into memory and PyTorch device
    """
    device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu',
    )
    kwargs: dict[str, Any] = {
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True,
    } if device == torch.device('cuda') else {}
    return kwargs, device


def label_change(
        data: ArrayLike,
        in_label: ArrayLike,
        one_hot: bool = False,
        out_label: ArrayLike | None = None) -> ArrayLike:
    """
    Converts an array or tensor of class values to an array or tensor of class indices

    Parameters
    ----------
    data : (N) ArrayLike
        Classes of size N
    in_label : (C) ArrayLike
        Unique class values of size C found in data
    one_hot : bool, default = False
        If the returned tensor should be 1D array of class indices or 2D one hot tensor if out_label
        is None or is an int
    out_label : (C) ArrayLike, default = None
        Unique class values of size C to transform data into, if None, then values will be indexes

    Returns
    -------
    (N) | (N,C) ArrayLike
        ndarray or Tensor of class indices, or if one_hot is True, one hot tensor
    """
    data_one_hot: ArrayLike
    out_data: ArrayLike
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
        out_label = out_label.to(data.device)

    assert out_label is not None
    out_data = out_label[module.searchsorted(in_label, data)]

    if one_hot:
        data_one_hot = module.zeros((len(data), len(in_label)))
        data_one_hot[module.arange(len(data)), out_data] = 1
        out_data = data_one_hot

    if isinstance(out_data, Tensor):
        out_data = out_data.to(data.device)

    return out_data


def progress_bar(i: int, total: int, text: str = '', **kwargs: Any) -> None:
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

    **kwargs
        Optional keyword arguments to pass to print
    """
    filled: int
    length: int = 50
    percent: float
    bar_fill: str
    i += 1

    filled = int(i * length / total)
    percent = i * 100 / total
    bar_fill = '█' * filled + '-' * (length - filled)
    print(f'\rProgress: |{bar_fill}| {int(percent)}%\t{text}\t', end='', **kwargs)

    if i == total:
        print()


def save_name(num: int | str, states_dir: str, name: str) -> str:
    """
    Standardises the network save file naming

    Parameters
    ----------
    num : int | str
        File number or name
    states_dir : str
        Directory of network saves
    name : str
        Name of the network

    Returns
    -------
    str
        Path to the network save file
    """
    return os.path.join(states_dir, f'{name}_{num}.pth')
