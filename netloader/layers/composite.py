"""
Layers made up of multiple layer types
"""
import torch
from torch import nn, Tensor


class Inception(nn.Module):
    """
    Creates the inception module and forward pass

    Attributes
    ----------
    elu : ELU
        ELU activation
    conv_1 : Conv2d | Conv1d
        Single convolutional layer with kernel 1 and 64 filters
    conv_11 : Conv2d | Conv1d
        First layer convolution with kernel 1 and 96 filters
    conv_12 : Conv2d | Conv1d
        First layer convolution with kernel 1 and 16 filters
    pool_11 : Pool2d | Pool1d
        First layer max pool with kernel 3
    conv_21 : Conv2d | Conv1d
        Second layer convolution with kernel 3 and 128 filters
    conv_22 : Conv2d | Conv1d
        Second layer convolution with kernel 5 and 32 filters
    conv_23 : Conv2d | Conv1d
        Second layer convolution with kernel 1 and 32 filters

    Methods
    -------
    forward(x)
        Forward pass of the inception module
    """
    def __init__(self, in_shape: list[int], dimension: bool = True):
        """
        Parameters
        ----------
        in_shape : list[integer]
            Input shape
        dimension : boolean, default = True
            If input is 2D or 1D
        """
        super().__init__()

        if dimension:
            conv = nn.Conv2d
            pool = nn.MaxPool2d
        else:
            conv = nn.Conv1d
            pool = nn.MaxPool1d

        self.elu = nn.ELU()
        self.conv_1 = conv(
            in_shape[0],
            64,
            1,
            padding='same',
            padding_mode='replicate',
        )
        self.conv_11 = conv(
            in_shape[0],
            96,
            1,
            padding='same',
            padding_mode='replicate',
        )
        self.conv_12 = conv(
            in_shape[0],
            16,
            1,
            padding='same',
            padding_mode='replicate',
        )
        self.pool_11 = pool(3, 1, 1)

        self.conv_21 = conv(
            96,
            128,
            3,
            padding='same',
            padding_mode='replicate',
        )
        self.conv_22 = conv(
            16,
            32,
            5,
            padding='same',
            padding_mode='replicate',
        )
        self.conv_23 = conv(
            in_shape[0],
            32,
            1,
            padding='same',
            padding_mode='replicate',
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the inception module

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Output tensor
        """
        x_1 = self.elu(self.conv_1(x))
        x_11 = self.elu(self.conv_11(x))
        x_12 = self.elu(self.conv_12(x))
        x_13 = self.pool_11(x)
        x_21 = self.elu(self.conv_21(x_11))
        x_22 = self.elu(self.conv_22(x_12))
        x_23 = self.elu(self.conv_23(x_13))

        return torch.cat((x_1, x_21, x_22, x_23))


def inception(kwargs: dict, layer: dict) -> dict:
    """
    Creates an inception layer

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_shape : list[integer]
            Shape of the outputs from each layer
        module : Sequential
            Sequential module to contain the layer;
    layer : dictionary
        2d : boolean, default = True
            If input is 2D or 1D

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['shape'].append(kwargs['shape'][-1].copy())
    kwargs['shape'][-1][0] = 256

    if ('2d' in layer and layer['2d']) or ('2d' not in layer and kwargs['2d']):
        dimension = True
    else:
        dimension = False

    kwargs['module'].add_module(
        f"inception_{kwargs['i']}",
        Inception(kwargs['shape'][-2], dimension=dimension),
    )

    return kwargs
