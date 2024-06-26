"""
Layer utility functions
"""
import logging as log
from typing import Any, Self

import torch
import numpy as np
from torch import nn, Tensor


class BaseLayer(nn.Module):
    """
    Base layer for other layers to inherit

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers part of the parent layer to loop through in the forward pass

    Methods
    -------
    initialise_layers()
        Checks if self.layers contains layers and exposes them to the optimiser so that weights can
        be trained
    forward(x, **_) -> Tensor
        Forward pass for a generic layer
    """
    def __init__(self, idx: int, **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number

        **kwargs
            Leftover parameters for checking if they are valid
        """
        super().__init__()
        self._device: torch.device = torch.device('cpu')
        self.layers: list[nn.Module] | nn.Sequential = []
        supported_params: list[str] = [
            'net_check',
            'net_out',
            'shapes',
            'check_shapes',
            'type',
            'group',
        ]
        keys: np.ndarray
        bad_params: np.ndarray

        if kwargs:
            keys = np.array(list(kwargs.keys()))
            bad_params = keys[~np.isin(keys, supported_params)]

            if len(bad_params):
                log.warning(
                    f'Unknown parameters for {self.__class__.__name__} in layer {idx}: {bad_params}'
                )

    def initialise_layers(self) -> None:
        """
        Checks if self.layers contains layers and exposes them to the optimiser so that weights can
        be trained
        """
        if isinstance(self.layers, list) and self.layers:
            self.layers = nn.Sequential(*self.layers)

    def forward(self, x: Tensor, **_: Any) -> Tensor:
        """
        Forward pass for a generic layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor with batch size N

        Returns
        -------
        (N,...) Tensor
            Output tensor with batch size N
        """
        assert isinstance(self.layers, nn.Sequential)
        return self.layers(x)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self._device, *_ = torch._C._nn._parse_to(*args, **kwargs)
        return self


class BaseMultiLayer(BaseLayer):
    """
    Base layer for layers that use earlier layers to inherit

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers part of the parent layer to loop through in the forward pass

    Methods
    -------
    extra_repr() -> string
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            net_check: bool,
            layer: int,
            shapes: list[list[int]],
            check_shapes: list[list[int]],
            checkpoint: bool = False,
            **kwargs: Any):
        """
        Parameters
        ----------
        net_check : boolean
            If layer index should be relative to checkpoint layers
        layer : integer
            Layer index to concatenate the previous layer output with
        shapes : list[list[integer]]
            Shape of the outputs from each layer
        check_shapes : list[list[integer]]
            Shape of the outputs from each checkpoint
        checkpoint : boolean, default = False
            If layer index should be relative to checkpoint layers

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._checkpoint: bool = checkpoint or net_check
        self._layer: int = layer
        self._target: list[int]

        # If checkpoints are being used
        if self._checkpoint:
            self._target = check_shapes[self._layer].copy()
        else:
            self._target = shapes[self._layer].copy()

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        string
            Layer parameters
        """
        return f'layer={self._layer}, checkpoint={bool(self._checkpoint)}'
