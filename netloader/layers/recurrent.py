"""
Recurrent network layers
"""
from typing import Any, Type

import torch
import numpy as np
from torch import nn, Tensor

from netloader.layers.misc import Reshape
from netloader.layers.base import BaseLayer


class Recurrent(BaseLayer):
    """
    Recurrent layer constructor for either RNN, GRU or LSTM

    Attributes
    ----------
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the recurrent layer
    extra_repr() -> string
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            idx: int,
            shapes: list[list[int]],
            batch_norm: bool = False,
            layers: int = 2,
            filters: int = 1,
            dropout: float = 0,
            mode: str = 'gru',
            activation: str | None = 'ELU',
            bidirectional: str | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        idx : integer
            Layer number
        shapes : list[list[integer]]
            Shape of the outputs from each layer
        batch_norm : boolean, default = False
            If batch normalisation should be used
        layers : integer, default = 2
            Number of stacked recurrent layers
        filters : integer, default = 1
            Number of output filters
        dropout : float, default =  0
            Probability of dropout, requires layers > 1
        mode : {'gru', 'rnn', 'lstm'}
            Type of recurrent layer
        activation : str | None, default = 'ELU'
            Which activation function to use, if mode is 'rnn', ReLU activation will always be used
        bidirectional : {None, 'sum', 'mean', 'concatenate'}
            If a bidirectional recurrence should be used and method for combining the two
            directions

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        self._bidirectional: str | None = bidirectional
        self._options: list[str | None] = [None, 'sum', 'mean', 'concatenate']
        shape: list[int] = [filters, int(np.prod(shapes[-1][1:]))]
        modes: dict[str, Type[nn.RNNBase]] = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        recurrent: nn.Module

        self._check_options('bidirectional', self._bidirectional, set(self._options))
        self._check_options('mode', mode, set(modes))

        if layers == 1:
            dropout = 0

        recurrent = modes[mode.lower()](
            input_size=shapes[-1][0],
            hidden_size=shape[0],
            num_layers=layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self._bidirectional is not None,
            **({'nonlinearity': 'relu'} if mode.lower() == 'rnn' else {}),
        )

        if bidirectional == 'concatenate':
            shape[0] *= 2

        # Convert 2D data to 1D and add recurrent layer
        self.layers.add_module('reshape', Reshape([shapes[-1][0], -1]))
        self.layers.add_module('recurrent', recurrent)

        # Optional layers
        if activation and mode.lower() != 'rnn':
            self.layers.add_module('activation', getattr(nn, activation)())

        if batch_norm:
            self.layers.add_module('batch_norm', nn.BatchNorm1d(shape[0]))

        shapes.append(shape)

    def forward(self, x: Tensor, **_: Any) -> Tensor:
        r"""
        Forward pass of the recurrent layer

        Parameters
        ----------
        x : `(N,C_{in},L)` Tensor
            Input tensor

        Returns
        -------
        `(N,C_{out},L)` Tensor
            Output tensor
        """
        x = self.layers[0](x)
        x = self.layers[1](torch.transpose(x, 1, 2))[0]

        if self._bidirectional in ['sum', 'mean']:
            x = x.view(*x.size()[:2], 2, -1)

            if self._bidirectional == self._options[1]:
                x = torch.sum(x, dim=-2)
            elif self._bidirectional == self._options[2]:
                x = torch.mean(x, dim=-2)

        x = torch.transpose(x, 1, 2)

        for layer in self.layers[2:]:
            x = layer(x)

        return x

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        string
            Layer parameters
        """
        return f'bidirectional_method={self._bidirectional}'
