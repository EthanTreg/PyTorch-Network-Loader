"""
Misc functions used elsewhere
"""
import os
import sys
import inspect
import logging as log
from types import ModuleType
from contextlib import contextmanager
from typing import Any, Literal, Generator, cast, overload

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from packaging.version import Version

from netloader.utils.types import ArrayT


class Shapes(list[list[int] | list[list[int]]]):
    """
    Shapes list that allows for indexing of multiple shapes.

    Methods
    -------
    check(idx) -> bool
        Check if the shape at the provided index is a single shape
    get(idx, list_=True) -> list[int] | list[list[int]] | Shapes
        Get shape from the shapes list
    """
    @overload  # type: ignore[override]
    def __getitem__(self, idx: int | tuple[int, int]) -> list[int]: ...

    @overload
    def __getitem__(self, idx: slice) -> 'Shapes': ...

    def __getitem__(self, idx: int | tuple[int, int] | slice) -> 'list[int] | Shapes':
        """
        Get shape from the shapes list.

        Parameters
        ----------
        idx : int | tuple[int, int] | slice
            Index of the shape to get, if tuple then first index is the layer index and second is
            the shape index if the layer has multiple shapes, if slice, returns a Shapes object with
            the sliced shapes

        Returns
        -------
        list[int] | Shapes
            Shape at the provided index
        """
        return self.get(idx, list_=False)

    @overload  # type: ignore[override]
    def __setitem__(self, idx: slice, value: list[list[int] | list[list[int]]]) -> None: ...

    @overload
    def __setitem__(self, idx: tuple[int, int], value: list[int]) -> None: ...

    @overload
    def __setitem__(self, idx: int, value: list[int] | list[list[int]]) -> None: ...

    def __setitem__(
            self,
            idx: int | tuple[int, int] | slice,
            value: list[int] | list[list[int]] | list[list[int] | list[list[int]]]) -> None:
        """
        Sets shape in the shapes list.

        Parameters
        ----------
        idx : int | tuple[int, int] | slice
            Index of the shape to set, if tuple then first index is the layer index and second is
            the shape index if the layer has multiple shapes, if slice, sets multiple shapes
        value : list[int] | list[list[int]]
            Shape to set at the provided index
        """
        if isinstance(idx, tuple):
            if not isinstance(self[idx[0]][0], list):
                raise ValueError(f'Index {idx} does not contain multiple shapes, shape at index '
                                 f'is {self[idx[0]]}')
            if not isinstance(value[0], int):
                raise ValueError(f'Index {idx} requires a single shape, got {value}')

            cast(list[list[int]], self[idx[0]])[idx[1]] = cast(list[int], value)
        elif isinstance(idx, int):
            if isinstance(value[0], list) and isinstance(value[0][0], list):
                raise ValueError(f'If index is an int, value must be a single shape or a list of '
                                 f'shapes, got {value}')
            super().__setitem__(idx, cast(list[int] | list[list[int]], value))
        else:
            if isinstance(value[0], int):
                raise ValueError(f'If index is a slice, value must be a list of shapes, got '
                                 f'{value}')
            super().__setitem__(idx, cast(list[list[int] | list[list[int]]], value))

    def __repr__(self) -> str:
        return f'Shapes({super().__repr__()})'

    def check(self, idx: int) -> bool:
        """
        Checks if the shape at the provided index is a single shape.

        Parameters
        ----------
        idx : int
            Index of the shape to check

        Returns
        -------
        bool
            True if the shape at the provided index is a single shape
        """
        if isinstance(self.get(idx, list_=True)[0], int):
            return True
        return False

    @overload
    def get(self, idx: slice, list_: bool) -> 'Shapes': ...

    @overload
    def get(self, idx: tuple[int, int], list_: bool) -> list[int]: ...

    @overload
    def get(self, idx: int, list_: Literal[False]) -> list[int]: ...

    @overload
    def get(self, idx: int, list_: Literal[True]) -> list[int] | list[list[int]]: ...

    @overload
    def get(self, idx: int, list_: bool) -> list[int] | list[list[int]]: ...

    def get(
            self,
            idx: int | tuple[int, int] | slice,
            list_: bool = True) -> 'list[int] | list[list[int]] | Shapes':
        """
        Get shape from the shapes list.

        Parameters
        ----------
        idx : int | tuple[int, int] | slice
            Index of the shape to get, if tuple then first index is the layer index and second is
            the shape index if the layer has multiple shapes, if slice, returns a Shapes object with
            the sliced shapes
        list_ : bool, default = True
            If true, returns a list of shapes if the index contains multiple shapes, else returns a
            single shape

        Returns
        -------
        list[int] | list[list[int]] | Shapes
            Shape at the provided index
        """
        item: list[int] | list[list[int]] | list[list[int] | list[list[int]]] = super().__getitem__(
            idx if isinstance(idx, (int, slice)) else idx[0],
        )

        if isinstance(idx, slice):
            return Shapes(cast(list[list[int] | list[list[int]]], item))
        if isinstance(idx, tuple) and isinstance(item[0], list):
            return cast(list[int], item[idx[1]])
        if isinstance(idx, tuple):
            raise ValueError(f'Index {idx} does not contain multiple shapes, shape at index is '
                             f'{item}')
        if isinstance(item[0], list) and not list_:
            raise ValueError(f'Index {idx} does not contain a single shape, shape at index is '
                             f'{item}, if you want to return all shapes at that index, use '
                             f'Shapes.get(idx, list_=True) instead')
        return cast(list[int], item)


def ascii_plot(
        data: list[float],
        *,
        clear: bool = False,
        x_num: int = 20,
        height: int = 6,
        text: str = '',
        label: str = '',
        data2: list[float] | None = None) -> None:
    """
    Prints a simple ASCII graph of data per index.

    Parameters
    ----------
    data : list[float]
        Data to plot
    clear : bool, default = False
        If true, clears the previous graph, WARNING: this is very buggy and may not work properly
    x_num : int, default = 20
        Maximum number of epochs to display in the graph
    height : int, default = 6
        Height of the graph in lines
    text : str, default = ''
        Optional text to display at the end of the graph
    label : str, default = ''
        Label for the x-axis
    data2 : list[float] | None, default = None
        Secondary data to plot
    """
    if len(data) < 2:
        return

    i: int
    row: int
    row2: int
    spacing: int = int(np.ceil(np.log10(len(data)))) + 1
    datum: float
    datum2: float | None
    line: str
    symbol: str
    symbol2: str
    x_label: str = f'{label}:' if label else ''
    grid: list[str]
    symbols: list[str] = ['_', '-', '‾']  # low, mid, high
    symbols2: list[str] = ['.', '·', '˙']
    rows: ndarray
    red_data2: ndarray
    red_data: ndarray = np.array(data)[-x_num:]

    if data2:
        red_data2 = np.array(data2)[-x_num:]
        range_ = (min(red_data.min(), red_data2.min()), max(red_data.max(), red_data2.max()))
        red_data2 = (red_data2 - range_[0]) / (range_[1] - range_[0])
    else:
        red_data2 = np.array([None] * len(red_data))
        range_ = (red_data.min(), red_data.max())

    rows = np.linspace(0, 1, height + 1)
    red_data = (red_data - range_[0]) / (range_[1] - range_[0])
    grid = [''] * height
    rows[-1] += 1

    if clear:
        sys.stdout.write(f'\033[{height + 2}F')

    for i, (datum, datum2) in enumerate(zip(red_data, red_data2)):
        row = int(np.digitize(datum, rows)) - 1
        symbol = symbols[min(
            int((datum - rows[row]) // (1 / len(symbols) / height)),
            len(symbols) - 1,
        )]

        if datum2 is not None:
            row2 = int(np.digitize(datum2, rows)) - 1
            symbol2 = symbols2[min(
                int((datum2 - rows[row2]) // (1 / len(symbols2) / height)),
                len(symbols2) - 1,
            )]

            if row == row2:
                symbol = '*'
            else:
                grid[row2] = f'{grid[row2]:<{i * spacing + len(x_label) + 1}}{symbol2 * spacing}'

        grid[row] = f'{grid[row]:<{i * spacing + len(x_label) + 1}}{symbol * spacing}'

    for line in grid[::-1]:
        print(line)

    print(
        x_label,
        ''.join(f'{i + 1:^{spacing}}' for i in range(max(0, len(data) - x_num), len(data))),
        f'\n{text}',
    )


def check_params(name: str, supported_params: list[str] | ndarray, in_params: ndarray) -> None:
    """
    Checks if provided parameters are supported by the function.

    Parameters
    ----------
    name : str
        Name of the function
    supported_params : list[str] | ndarray
        Parameters supported by the function
    in_params : ndarray
        Input parameters of shape (N), where N is the number of parameters
    """
    bad_params: ndarray = in_params[~np.isin(in_params, supported_params)]

    if len(bad_params):
        log.getLogger(__name__).warning(f'Unknown parameters for {name}: {bad_params}')


def compare_versions(cur_ver: str, min_ver: str) -> bool:
    """
    Compares two version strings.

    Parameters
    ----------
    cur_ver : str
        Current version
    min_ver : str
        Minimum version

    Returns
    -------
    bool
        True if current version is greater than or equal to minimum version
    """
    if not bool(cur_ver):
        return True
    return not '<' in cur_ver and Version(cur_ver) >= Version(min_ver)


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


def dict_list_append(
        dict1: dict[str, Any],
        dict2: dict[str, Any],
        concat: bool = False,
        balance: bool = False) -> dict[str, Any]:
    """
    Merges two dictionaries.
    If a key exists in both dictionaries and not concat, the data type must match.
    If data type is list or ndarray, the values are extended.
    If data type is any other type and not concat, the value from the second dictionary overrides
    the first, else if concat, the values are converted to a list and appended.
    If balance is True, all lists and ndarrays in each dictionary must have the same length, if a
    key is missing from one dictionary, it is added with None values to match the length of the
    other lists/ndarrays in that dictionary.

    Parameters
    ----------
    dict1 : dict[str, Any]
        Primary dict to merge secondary dict into
    dict2 : dict[str, Any]
        Secondary dict to merge into primary dict
    concat : bool, default = False
        If true, concatenates non list/ndarray types instead of overriding them
    balance : bool, default = False
        If true, balances the lengths of lists/ndarrays in each dictionary so that each list/ndarray
        in the returning dictionary has the same length

    Returns
    -------
    dict[str, Any]
        First dict with second dict merged into it
    """
    dict1_len: int = 0
    dict2_len: int = 0
    key: str

    if not concat:
        for key in np.intersect1d(list(dict1.keys()), list(dict2.keys())).tolist():
            if not isinstance(dict1[key], type(dict2[key])):
                raise TypeError(f'Type mismatch for key "{key}": {type(dict1[key])} != '
                                f'{type(dict2[key])}')

    if balance:
        # If primary dict is not empty, find the length of a list/ndarray in the dictionary
        if len(dict1) > 0:
            for value in dict1.values():
                if isinstance(value, (list, ndarray)):
                    dict1_len = len(value)
                    break

        if any(len(value) != dict1_len for value in dict1.values()
               if isinstance(value, (list, ndarray))):
            raise ValueError('All lists/ndarrays in dict1 must have the same length')

        # If secondary dict is not empty, find the length of a list/ndarray in the dictionary
        if len(dict2) > 0:
            for value in dict2.values():
                if isinstance(value, (list, ndarray)):
                    dict2_len = len(value)
                    break

        if any(len(value) != dict2_len for value in dict2.values()
               if isinstance(value, (list, ndarray))):
            raise ValueError('All lists/ndarrays in dict2 must have the same length')

    # Merge two dictionaries
    for key in np.unique(list(dict1.keys()) + list(dict2.keys())).tolist():
        # If the secondary dict has a key not in the primary, pad with Nones
        if key in dict1 or not balance:
            pass
        elif isinstance(dict2[key], ndarray):
            dict1[key] = np.full(
                [dict1_len] + (list(dict2[key][0].shape) if dict2[key].ndim > 1 else []),
                None,
            )
        elif isinstance(dict2[key], list) or concat:
            dict1[key] = [None] * dict1_len

        if concat and key in dict1 and not isinstance(dict1[key], (list, ndarray)):
            dict1[key] = [dict1[key]]

        if concat and key in dict2 and not isinstance(dict2[key], (list, ndarray)):
            dict2[key] = [dict2[key]]

        # If the primary dict has a key not in the secondary dict, pad with Nones, else merge dicts
        if key not in dict2 and not balance:
            continue
        if key not in dict1:
            dict1[key] = dict2[key]
        elif balance and key not in dict2 and isinstance(dict1[key], ndarray):
            dict1[key] = np.concat(
                (dict1[key], [np.full(  # type: ignore[arg-type]
                    dict1[key][0].shape,
                    None,
                ) if dict1[key].ndim > 1 else None] * dict2_len),
                axis=0,
            )
        elif balance and key not in dict2 and isinstance(dict1[key], list):
            dict1[key].extend([None] * dict2_len)
        elif isinstance(dict1[key], ndarray):
            dict1[key] = np.concatenate((dict1[key], dict2[key]), axis=0)
        elif isinstance(dict1[key], list):
            dict1[key].extend(dict2[key])
        elif key in dict2:
            dict1[key] = dict2[key]
    return dict1


def dict_list_convert(data: dict[str, list[Any] | ndarray]) -> list[dict[str, Any]]:
    """
    Converts a dictionary of lists to a list of dictionaries.

    Parameters
    ----------
    data : dict[str, list[Any] | ndarray]
        Dictionary of lists to convert

    Returns
    -------
    list[dict[str, Any]]
        List of dictionaries
    """
    return [{key: data[key][i:i + 1] for key in data} for i in range(len(list(data.values())[0]))]


def get_device() -> tuple[dict[str, Any], torch.device]:
    """
    Gets the device for PyTorch to use.

    Returns
    -------
    tuple[dict[str, Any], torch.device]
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
        data: ArrayT,
        in_label: ArrayT,
        *,
        one_hot: bool = False,
        out_label: ArrayT | None = None) -> ArrayT:
    """
    Converts an array or tensor of class values to an array or tensor of class indices.

    Parameters
    ----------
    data : ArrayT
        Classes of shape (N) where N is the number of samples
    in_label : ArrayT
        Unique class values of size (C) where C is the number of classes
    one_hot : bool, default = False
        If the returned tensor should be 1D array of class indices or 2D one hot tensor if out_label
        is None or is an int
    out_label : ArrayT | None, default = None
        Unique class values of size (C) to transform data into, if None, then values will be indexes

    Returns
    -------
    ArrayT
        ndarray or Tensor of class indices of shape (N) and type float, or if one_hot is True, one
        hot tensor of shape (N,C) and type float
    """
    data_one_hot: ArrayT
    out_data: ArrayT
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
        out_label = cast(ArrayT, out_label.to(data.device))

    out_data = cast(ArrayT, out_label[module.searchsorted(in_label, data)])

    if one_hot:
        data_one_hot = module.zeros((len(data), len(in_label)))
        data_one_hot[module.arange(len(data)), out_data] = 1
        out_data = data_one_hot

    return cast(ArrayT, out_data.to(data.device) if isinstance(out_data, Tensor) else out_data)


def list_dict_convert(
        data: list[dict[str, Any]],
        balance: bool = False,
        concat: bool = False) -> dict[str, list[Any]]:
    """
    Converts a list of dictionaries to a dictionary of lists

    Parameters
    ----------
    data : list[dict[str, float | ndarray]]
        List of dictionaries to convert
    balance : bool, default = False
        If true, balances the lengths of lists/ndarrays in each dictionary so that each list/ndarray
        in the returning dictionary has the same length
    concat : bool, default = False
        If true, concatenates non list/ndarray types instead of overriding them

    Returns
    -------
    dict[str, list[float | None] | list[ndarray]]
        Dictionary of lists
    """
    value: dict[str, float | ndarray]
    new_data: dict[str, list[float] | list[ndarray]] = {}

    for value in data:
        dict_list_append(new_data, value, concat=concat, balance=balance)
    return new_data


def progress_bar(i: int, total: int, *, text: str = '', **kwargs: Any) -> None:
    """
    Terminal progress bar.

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


def safe_globals(safe_package: str, modules: ModuleType | list[ModuleType]) -> None:
    """
    Adds all classes in the module from the specified package to the list of safe PyTorch classes
    when loading saved networks.

    Parameters
    ----------
    safe_package : str
        Package that classes must belong to to be added to the safe globals
    modules : ModuleType | list[ModuleType]
        Module(s) to add all classes from to the list of safe PyTorch classes when loading saved
        networks
    """
    if isinstance(modules, ModuleType):
        modules = [modules]

    for module in modules:
        torch.serialization.add_safe_globals([
            member[1] for member in inspect.getmembers(sys.modules[module.__name__])
            if isinstance(member[1], type) and
               member[1].__module__.split('.', maxsplit=1)[0] == safe_package
        ])


def save_name(num: int | str, states_dir: str, name: str) -> str:
    """
    Standardises the network save file naming.

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

@contextmanager
def suppress_logger_warnings(logger_name: str) -> Generator[None, None, None]:
    """
    Suppresses warnings from a specified logger.

    Parameters
    ----------
    logger_name : str
        Name of the logger to suppress warnings from
    """
    level: int
    logger: log.Logger = log.getLogger(logger_name)
    level = logger.level
    logger.setLevel(log.ERROR)

    try:
        yield
    finally:
        logger.setLevel(level)
