"""
Recurrent network layers
"""
import torch
from torch import nn, Tensor

from netloader.layers.utils import optional_layer


class RecurrentOutput(nn.Module):
    """
    GRU wrapper for compatibility with network & can handle output of bidirectional GRUs

    Attributes
    ----------
    bidirectional : string, default = None
        If GRU is bidirectional, and if so, what method to use,
        can be either mean, sum or concatenation,
        if None, GRU is mono-directional and concatenate will be used

    Methods
    -------
    forward(x)
        Returns
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
        Forward pass of the GRU

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


def recurrent(kwargs: dict, layer: dict) -> dict:
    """
    Recurrent layer constructor for either RNN, GRU or LSTM

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or
            convolutional/recurrent filters;
        module : Sequential
            Sequential module to contain the layer;
        dropout_prob : float, optional
            Probability of dropout, only required if layers from layer > 1;
    layer : dictionary
        dropout : boolean, default = True
            If dropout should be used;
        batch_norm : boolean, default = 0
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;
        layers : integer, default = 2
            Number of stacked GRU layers;
        filters : integer, default = 1
            Number of output filters;
        method : string, default = gru
            Type of recurrent layer, can be gru, lstm or rnn;
        bidirectional : string, default = None
            If a bidirectional recurrence should be used and
            method for combining the two directions, can be sum, mean or concatenation;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    try:
        layers = layer['layers']
    except KeyError:
        layers = 2

    try:
        kwargs['dims'].append(layer['filters'])
    except KeyError:
        kwargs['dims'].append(1)

    try:
        bidirectional = layer['bidirectional']

        if bidirectional == 'None':
            bidirectional = None
    except KeyError:
        bidirectional = None

    if layers > 1 and (('dropout' in layer and layer['dropout']) or 'dropout' not in layer):
        dropout_prob = kwargs['dropout_prob']
    else:
        dropout_prob = 0

    recurrent_kwargs = {
        'input_size': kwargs['dims'][-2],
        'hidden_size': kwargs['dims'][-1],
        'num_layers': layers,
        'batch_first': True,
        'dropout': dropout_prob,
        'bidirectional': bidirectional is not None,
    }

    if bidirectional == 'concatenate':
        kwargs['dims'][-1] *= 2

    try:
        if layer['method'] == 'rnn':
            recurrent_layer = nn.RNN(**recurrent_kwargs, nonlinearity='relu')
        elif layer['method'] == 'lstm':
            recurrent_layer = nn.LSTM(**recurrent_kwargs)
        else:
            recurrent_layer = nn.GRU(**recurrent_kwargs)
    except KeyError:
        recurrent_layer = nn.GRU(**recurrent_kwargs)

    kwargs['module'].add_module(
        f"GRU_output_{kwargs['i']}",
        RecurrentOutput(recurrent_layer, bidirectional=bidirectional)
    )

    optional_layer(False, 'batch_norm', kwargs, layer, nn.BatchNorm1d(kwargs['dims'][-1]))
    optional_layer(True, 'activation', kwargs, layer, nn.ELU())

    kwargs['data_size'].append(kwargs['data_size'][-1])

    return kwargs