"""
Recurrent network layers
"""
from typing import Any, Type, Literal

import torch
import numpy as np
from torch import nn, Tensor

from netloader.utils import Shapes
from netloader.layers.misc import Reshape
from netloader.layers.base import BaseSingleLayer


class Recurrent(BaseSingleLayer):
    """
    Recurrent layer constructor for either RNN, GRU or LSTM.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : nn.Sequential
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
            shapes: Shapes,
            *,
            batch_norm: bool = False,
            layers: int = 2,
            filters: int = 1,
            dropout: float = 0,
            mode: Literal['gru', 'lstm', 'rnn'] = 'gru',
            activation: str | None = 'ELU',
            bidirectional: Literal[None, 'sum', 'mean', 'concatenate'] = None,
            **kwargs: Any) -> None:
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
        modes: dict[str, Type[nn.RNN] | Type[nn.LSTM] | Type[nn.GRU]] = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU,
        }
        recurrent: nn.RNNBase

        self._check_options('bidirectional', self._bidirectional, set(self._options))
        self._check_options('mode', mode, set(modes))

        if layers == 1:
            dropout = 0

        recurrent = modes[mode.lower()](
            shapes[-1][0],
            shape[0],
            num_layers=layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=self._bidirectional is not None,
            **({'nonlinearity': 'relu'} if mode.lower() == 'rnn' else {}),
        )

        if bidirectional == 'concatenate':
            shape[0] *= 2

        # Convert 2D data to 1D and add recurrent layer
        self.layers.add_module('Reshape', Reshape([shapes[-1][0], -1]))
        self.layers.add_module('Recurrent', recurrent)

        # Optional layers
        if activation and mode.lower() != 'rnn':
            self.layers.add_module('Activation', getattr(nn, activation)())

        if batch_norm:
            self.layers.add_module('BatchNorm', nn.BatchNorm1d(shape[0]))

        shapes.append(shape)

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'batch_norm': hasattr(self.layers, 'BatchNorm'),
            'layers': self.layers.Recurrent.num_layers,  # type: ignore[union-attr]
            'filters': self.layers.Recurrent.hidden_size,  # type: ignore[union-attr]
            'dropout': self.layers.Recurrent.dropout,  # type: ignore[union-attr]
            'mode': self.layers.Recurrent.__class__.__name__.lower(),
            'activation': self.layers.Activation.__class__.__name__
                if hasattr(self.layers, 'Activation') else None,
            'bidirectional': self._bidirectional,
        }

    def forward(self, x: Tensor, *_: Any, **__: Any) -> Tensor:
        r"""
        Forward pass of the recurrent layer

        Parameters
        ----------
        x : Tensor
            Input tensor with shape `(N,C_{in},L)` and type float, where N is the batch size,
            `C_{in}` is the number of input channels and L is the length of the sequence

        Returns
        -------
        Tensor
            Output tensor with shape `(N,C_{out},L)` and type float, where `C_{out}` is the number
            of output channels
        """
        x = self.layers[0](x)
        x = self.layers[1](torch.transpose(x, 1, 2))[0]

        if self._bidirectional in ['sum', 'mean']:
            x = x.view(*x.size()[:2], 2, -1)

            if self._bidirectional == self._options[1]:
                x = torch.sum(x, dim=-2)
            elif self._bidirectional == self._options[2]:
                x = torch.mean(x, dim=-2)
        return self.layers[2:](torch.transpose(x, 1, 2))

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        string
            Layer parameters
        """
        return f'bidirectional_method={self._bidirectional}'
