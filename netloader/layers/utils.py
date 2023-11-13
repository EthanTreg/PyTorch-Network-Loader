"""
Layer utility functions
"""
import logging as log

import numpy as np
from torch import nn


def optional_layer(
        default: bool,
        arg: str,
        kwargs: dict,
        layer: dict,
        layer_func: nn.Module):
    """
    Implements an optional layer for a parent layer to use

    Parameters
    ----------
    default : boolean
        If the layer should be used by default
    arg : string
        Argument for the user to call this layer
    kwargs : dictionary
        kwargs dictionary used by the parent
    layer : dictionary
        layer dictionary used by the parent
    layer_func : Module
        Optional layer to add to the network
    """
    if (arg in layer and layer[arg]) or (arg not in layer and default):
        kwargs['module'].add_module(f"{type(layer_func).__name__}_{kwargs['i']}", layer_func)


def check_layer(layer_num: int, supported_params: list[str], layer: dict):
    """
    Checks if any layer parameters are unknown

    Parameters
    ----------
    layer_num : integer
        Layer number
    supported_params : list[string]
        List of parameters used in the layer
    layer : dictionary
        Argument with parameters passed to the layer to check
    """
    keys = np.array(list(layer.keys()))
    supported_params.append('type')

    bad_params = np.argwhere(~np.isin(keys, supported_params)).ravel()

    if keys[bad_params]:
        log.warning(f'Unknown parameters in layer {layer_num}: {keys[bad_params]}')
