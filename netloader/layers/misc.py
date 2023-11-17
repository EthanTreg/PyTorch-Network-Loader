"""
Miscellaneous network layers
"""
import numpy as np
from torch import nn, Tensor

from netloader.layers.utils import check_layer


class Reshape(nn.Module):
    """
    Used for reshaping tensors within a neural network

    Attributes
    shape : list[integer]
        Desired shape of the output tensor, ignoring first dimension

    Methods
    -------
    forward(x)
        Forward pass of Reshape
    """
    def __init__(self, shape: list[int]):
        """
        Parameters
        ----------
        shape : list[integer]
            Desired shape of the output tensor, ignoring first dimension
        """
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of reshaping tensors

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Output tensor
        """
        return x.contiguous().view(x.size(0), *self.shape)


class Index(nn.Module):
    """
    Indexes tensor using slicing

    Attributes
    ----------
    number : integer
        Number of values to slice, can be negative
    greater : boolean, default = True
        If slicing should include all values greater or less than number index

    Methods
    -------
    forward(x)
        Forward pass of the indexing layer
    """
    def __init__(self, number: int, greater: bool = True):
        """
        Parameters
        ----------
        number : integer
            Number of values to slice, can be negative
        greater : boolean, default = True
            If slicing should include all values greater or less than number index
        """
        super().__init__()
        self.number = number
        self.greater = greater

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the indexing layer

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Output tensor
        """
        if self.greater:
            return x[..., self.number:]

        return x[..., :self.number]


def clone(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Constructs a layer to clone a number of values from the previous layer

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
    layer : dictionary
        number : integer
            Number of values to clone from the previous layer
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    check_layer(['number'], kwargs, layer, check_params=check_params)
    kwargs['shape'].append(kwargs['shape'][-1].copy())


def concatenate(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Constructs a concatenation layer to combine the outputs from two layers

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
    layer : dictionary
        layer : integer
            Layer index to concatenate the previous layer output with
        dim : integer, default = 0
            Dimension to concatenate to
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    supported_params = ['layer', 'dim']
    layer = check_layer(supported_params, kwargs, layer, check_params=check_params)

    shape = kwargs['shape'][-1].copy()
    in_shape = shape.copy()
    target_shape = kwargs['shape'][layer['layer']].copy()
    dim = layer['dim']

    if len(target_shape) > 1 and len(in_shape) > 1:
        del target_shape[dim]
        del in_shape[dim]

    # If tensors cannot be concatenated along the specified dimension
    if ((target_shape != in_shape and len(target_shape) != 1) or
            (len(target_shape) != len(in_shape))):
        raise ValueError(
            f"Shape mismatch, input shape {shape} in layer {kwargs['i']} does not match the "
            f"target shape {kwargs['shape'][layer['layer']]} in layer {layer['layer']} "
            f"for concatenation over dimension {dim}"
        )

    shape[dim] = kwargs['shape'][-1][dim] + kwargs['shape'][layer['layer']][dim]
    kwargs['shape'].append(shape)


def extract(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Extracts a number of values from the last dimension, returning two tensors

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
    layer : dictionary
        number : integer
            Number of values to extract from the previous layer
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    layer = check_layer(['number'], kwargs, layer, check_params=check_params)
    kwargs['shape'].append(kwargs['shape'][-1].copy())
    kwargs['shape'][-1][-1] -= layer['number']


def index(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Constructs a layer to slice the last dimension from the output from the previous layer

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
        number : integer
            Index slice number
        greater : boolean, default = True
            If slice should be values greater or less than number
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    supported_params = ['number', 'greater']
    layer = check_layer(supported_params, kwargs, layer, check_params=check_params)
    kwargs['shape'].append(kwargs['shape'][-1].copy())

    # Length of slice
    if 'greater' in layer and (
            (layer['greater'] and layer['number'] < 0) or
            (not layer['greater'] and layer['number'] > 0)
    ):
        kwargs['shape'][-1][-1] = abs(layer['number'])
    else:
        kwargs['shape'][-1][-1] = kwargs['shape'][-1][-1] - abs(layer['number'])

    kwargs['module'].add_module(
        f"index_{kwargs['i']}",
        Index(layer['number'], greater=layer['greater']),
    )


def reshape(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Constructs a reshaping layer to change the data dimensions

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
        output : integer | list[integer]
            Output dimensions of input tensor, ignoring the first dimension (batch size)
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    layer = check_layer(['output'], kwargs, layer, check_params=check_params)

    # If -1 in output shape, calculate the dimension length from the input dimensions
    if -1 not in layer['output']:
        kwargs['shape'].append(layer['output'])
    elif layer['output'].count(-1) == 1:
        shape = layer['output'].copy()
        fixed_shape = np.delete(shape, np.array(shape) == -1)
        shape[shape.index(-1)] = int(np.prod(kwargs['shape'][-1]) / np.prod(fixed_shape))
        kwargs['shape'].append(shape)
    else:
        raise ValueError(
            f"Cannot infer output shape as -1 occurs more than once in {layer['output']}"
        )

    # If input tensor cannot be reshaped into output shape
    if np.prod(kwargs['shape'][-1]) != np.prod(kwargs['shape'][-2]):
        raise ValueError(
            f"Output size does not match input size"
            f"for input shape {kwargs['shape'][-2]} and output shape {kwargs['shape'][-1]}"
        )

    kwargs['module'].add_module(f"reshape_{kwargs['i']}", Reshape(layer['output']))


def shortcut(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Constructs a shortcut layer to add the outputs from two layers

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
    layer : dictionary
        layer : integer
            Layer index to add the previous layer output with
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    layer = check_layer(['layer'], kwargs, layer, check_params=check_params)
    in_shape = np.array(kwargs['shape'][-1].copy())
    target_shape = np.array(kwargs['shape'][layer['layer']].copy())
    mask = (in_shape != 1) & (target_shape != 1)

    if not np.array_equal(in_shape[mask], target_shape[mask]):
        raise ValueError(
            f"Tensor shapes {kwargs['shape'][-1]} in layer {kwargs['i']} and "
            f"{kwargs['shape'][layer['layer']]} in layer {layer['layer']} "
            f"not compatible for addition."
        )

    # If input has any dimensions of length one, output will take the target dimension
    if 1 in in_shape:
        idxs = np.where(in_shape == 1)[0]
        in_shape[idxs] = target_shape[idxs]

    # If target has any dimensions of length one, output will take the input dimension
    if 1 in target_shape:
        idxs = np.where(target_shape == 1)[0]
        in_shape[idxs] = in_shape[idxs]

    kwargs['shape'].append(in_shape.tolist())


def skip(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Bypasses previous layers by retrieving the output from the defined layer

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
    layer : dictionary
        layer : integer
            Layer index to retrieve the output
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    layer = check_layer(['layer'], kwargs, layer, check_params=check_params)
    kwargs['shape'].append(kwargs['shape'][layer['layer']].copy())
