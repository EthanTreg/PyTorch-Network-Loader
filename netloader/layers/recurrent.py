"""
Recurrent network layers
"""
import torch
import numpy as np
from torch import nn, Tensor

from netloader.layers.misc import Reshape
from netloader.layers.utils import optional_layer, check_layer


class Recurrent(nn.Module):
    """
    Recurrent wrapper for compatibility with network & can handle output of bidirectional RNNs

    Attributes
    ----------
    bidirectional : string, default = None
        If GRU is bidirectional, and if so, what method to use,
        can be either mean, sum or concatenation,
        if None, GRU is mono-directional and concatenate will be used

    Methods
    -------
    forward(x)
        Forward pass of the recurrent layer
    """
    def __init__(self, recurrent_layer: nn.Module, bidirectional: str = None):
        """
        Parameters
        ----------
        bidirectional : string, default = None
            If GRU is bidirectional, and if so, what method to use,
            can be either mean, sum or concatenate,
            if None, GRU is mono-directional and concatenation will be used
        """
        super().__init__()
        self.options = [None, 'sum', 'mean', 'concatenate']
        self.bidirectional = bidirectional
        self.recurrent_layer = recurrent_layer

        if self.bidirectional not in self.options:
            raise ValueError(
                f'{self.bidirectional} is not a valid bidirectional method, options: {self.options}'
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the recurrent layer

        Parameters
        ----------
        x : Tensor, shape (N, 1, L)
            Input tensor

        Returns
        -------
        Tensor, shape (N, 1, L)
            Output tensor
        """
        x = self.recurrent_layer(torch.transpose(x, 1, 2))[0]

        if self.bidirectional and self.bidirectional != self.options[3]:
            x = x.view(*x.size()[:2], 2, -1)

            if self.bidirectional == self.options[1]:
                x = torch.sum(x, dim=-2)
            elif self.bidirectional == self.options[2]:
                x = torch.mean(x, dim=-2)

        return torch.transpose(x, 1, 2)


def recurrent(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Recurrent layer constructor for either RNN, GRU or LSTM

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
        module : Sequential
            Sequential module to contain the layer
    layer : dictionary
        dropout : float, default =  0.1
            Probability of dropout
        batch_norm : boolean, default = 0
            If batch normalisation should be used
        activation : boolean, default = True
            If ELU activation should be used
        layers : integer, default = 2
            Number of stacked GRU layers
        filters : integer, default = 1
            Number of output filters
        method : string, default = gru
            Type of recurrent layer, can be gru, lstm or rnn
        bidirectional : string, default = None
            If a bidirectional recurrence should be used and
            method for combining the two directions, can be sum, mean or concatenation
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    supported_params = [
        'dropout',
        'batch_norm',
        'activation',
        'layers',
        'filters',
        'method',
        'bidirectional',
    ]
    layer = check_layer(supported_params, kwargs, layer, check_params=check_params)

    kwargs['shape'].append([kwargs['shape'][-1][0], np.prod(kwargs['shape'][-1][1:])])
    layers = layer['layers']
    kwargs['shape'][-1][0] = layer['filters']
    bidirectional = layer['bidirectional']

    if layers == 1:
        dropout = 0
    else:
        dropout = layer['dropout']

    recurrent_kwargs = {
        'input_size': kwargs['shape'][-2][0],
        'hidden_size': kwargs['shape'][-1][0],
        'num_layers': layers,
        'batch_first': True,
        'dropout': dropout,
        'bidirectional': bidirectional is not None,
    }

    if bidirectional == 'concatenate':
        kwargs['shape'][-1][0] *= 2

    if layer['method'] == 'rnn':
        recurrent_layer = nn.RNN(**recurrent_kwargs, nonlinearity='relu')
    elif layer['method'] == 'lstm':
        recurrent_layer = nn.LSTM(**recurrent_kwargs)
    else:
        recurrent_layer = nn.GRU(**recurrent_kwargs)

    # Convert 2D data to 1D and add recurrent layer
    kwargs['module'].add_module(f"reshape_{kwargs['i']}", Reshape([kwargs['shape'][-2][0], -1]))
    kwargs['module'].add_module(
        f"recurrent_{kwargs['i']}",
        Recurrent(recurrent_layer, bidirectional=bidirectional)
    )

    # Optional layers
    optional_layer('batch_norm', kwargs, layer, nn.BatchNorm1d(kwargs['shape'][-1][0]))
    optional_layer('activation', kwargs, layer, nn.ELU())

    return kwargs
