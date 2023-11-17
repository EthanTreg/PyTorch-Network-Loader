"""
Layer utility functions
"""
import logging as log

import numpy as np
from torch import nn


def optional_layer(
        name: str,
        kwargs: dict,
        layer: dict,
        layer_func: nn.Module):
    """
    Implements an optional layer for a parent layer to use

    Parameters
    ----------
    name : string
        Layer name
    kwargs : dictionary
        kwargs dictionary used by the parent
    layer : dictionary
        layer dictionary used by the parent
    layer_func : Module
        Optional layer to add to the network
    """
    if layer[name]:
        kwargs['module'].add_module(f"{name}_{kwargs['i']}", layer_func)


def check_layer(
        supported_params: list[str],
        kwargs: dict,
        layer: dict,
        check_params: bool = True) -> dict:
    """
    Checks if any layer parameters are unknown and merges default values

    Parameters
    ----------
    supported_params : list[string]
        List of parameters used in the layer
    kwargs : dictionary
        Network parameters with layer defaults to merge
    layer : dictionary
        Argument with parameters passed to the layer to check
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid

    Returns
    -------
    dictionary
        Layer parameters with defaults if values not provided
    """
    if check_params:
        keys = np.array(list(layer.keys()))
        supported_params.append('type')

        bad_params = np.argwhere(~np.isin(keys, supported_params)).ravel()

        if keys[bad_params]:
            log.warning(
                f"Unknown parameters for {layer['type']} in layer {kwargs['i']}: {keys[bad_params]}"
            )

    return kwargs[layer['type']] | layer
