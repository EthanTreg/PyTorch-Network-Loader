"""
Recurrent network layers
"""
from typing import Any

import torch
import numpy as np
from torch import nn, Tensor

from netloader.layers.misc import Reshape
from netloader.layers.utils import BaseLayer


class Recurrent(BaseLayer):
    """
    Recurrent layer constructor for either RNN, GRU or LSTM

    Attributes
    ----------
    layers : list[Module] | Sequential
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
            activation: bool = True,
            layers: int = 2,
            filters: int = 1,
            dropout: float = 0.1,
            method: str = 'gru',
            bidirectional: str | None = None,
            **kwargs):
        """
        Parameters
        ----------
        idx : integer
            Layer number
        shapes : list[list[integer]]
            Shape of the outputs from each layer
        batch_norm : boolean, default = False
            If batch normalisation should be used
        activation : boolean, default = True
            If ELU activation should be used
        layers : integer, default = 2
            Number of stacked recurrent layers
        filters : integer, default = 1
            Number of output filters
        dropout : float, default =  0.1
            Probability of dropout, requires layers > 1
        method : {'gru', 'rnn', 'lstm'}
            Type of recurrent layer
        bidirectional : {None, 'sum', 'mean', 'concatenate'}
            If a bidirectional recurrence should be used and method for combining the two
            directions

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        self._activation: bool
        self._batch_norm: bool
        self._dropout: float
        self._method: str
        self._bidirectional: str | None
        self._options: list[str | None]
        shape: list[int]
        recurrent_kwargs: dict[str, Any]
        recurrent: nn.Module

        self._activation = activation
        self._batch_norm = batch_norm
        self._dropout = dropout
        self._method = method
        self._bidirectional = bidirectional
        self._options = [None, 'sum', 'mean', 'concatenate']
        shape = [filters, np.prod(shapes[-1][1:])]

        if self._bidirectional not in self._options:
            raise ValueError(f'{self._bidirectional} in layer {idx} is not a valid bidirectional '
                             f'method, valid options are: {self._options}')

        if layers == 1:
            self._dropout = 0

        recurrent_kwargs = {
            'input_size': shapes[-1][0],
            'hidden_size': shape[0],
            'num_layers': layers,
            'batch_first': True,
            'dropout': self._dropout,
            'bidirectional': self._bidirectional is not None,
        }

        if bidirectional == 'concatenate':
            shape[0] *= 2

        if self._method == 'rnn':
            recurrent = nn.RNN(**recurrent_kwargs, nonlinearity='relu')
        elif self._method == 'lstm':
            recurrent = nn.LSTM(**recurrent_kwargs)
        else:
            recurrent = nn.GRU(**recurrent_kwargs)

        # Convert 2D data to 1D and add recurrent layer
        self.layers.append(Reshape([shapes[-1][0], -1]))
        self.layers.append(recurrent)

        # Optional layers
        if self._activation and self._method != 'rnn':
            self.layers.append(nn.ELU())

        if self._batch_norm:
            self.layers.append(nn.BatchNorm1d(shape[0]))

        shapes.append(shape)

    def forward(self, x: Tensor, **_) -> Tensor:
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
        return f'bidirectional_method={self._bidirectional}, activation={bool(self._activation)}'
