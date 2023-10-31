"""
Miscellaneous network layers
"""
import torch
from torch import nn, Tensor


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


class Squeeze(nn.Module):
    """
    Squeezes or un-squeezes a dimension

    Attributes
    ----------
    remove : boolean
        If dimension should be squeezed or un-squeezed
    dim : integer
        Target dimension

    Methods
    -------
    forward(x)
        Forward pass of squeeze
    """
    def __init__(self, remove: bool, dim: int):
        """
        Parameters
        ----------
        remove : boolean
            If dimension should be squeezed or un-squeezed
        dim : integer
            Target dimension
        """
        super().__init__()
        self.remove = remove
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of squeeze

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Output tensor
        """
        if self.remove:
            x = torch.squeeze(x, self.dim)
        else:
            x = torch.unsqueeze(x, self.dim)

        return x


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


def reshape(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a reshaping layer to change the data dimensions

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    layer : dictionary
        output : integer | list[integer]
            Output dimensions of input tensor, ignoring the first dimension (batch size) and
            subsequent dimensions if the number of dimensions in output is less than the dimensions
            of the input tensor, if output = -1, then last two dimensions are flattened

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    # If reshape reduces the number of dimensions
    if len(layer['output']) == 1:
        kwargs['dims'].append(kwargs['data_size'][-1] * kwargs['dims'][-1])

        # Data size equals the previous size multiplied by the previous dimension
        kwargs['data_size'].append(kwargs['dims'][-1])
    else:
        kwargs['dims'].append(layer['output'][0])

        # Data size equals the previous size divided by the first shape dimension
        kwargs['data_size'].append(int(kwargs['data_size'][-1] / kwargs['dims'][-1]))

    kwargs['module'].add_module(f"reshape_{kwargs['i']}", Reshape(layer['output']))

    return kwargs


def squeeze(kwargs: dict, layer: dict) -> dict:
    """
    Squeezes or un-squeezes a dimension of the tensor

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    layer : dictionary
        squeeze : boolean
            If dimension should be removed (True) or added (False);
        dim : integer
            Which dimension should be edited

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['module'].add_module(f"squeeze_{kwargs['i']}", Squeeze(layer['squeeze'], layer['dim']))
    kwargs['data_size'].append(kwargs['data_size'][-1])

    if layer['squeeze']:
        kwargs['dims'].append(kwargs['data_size'][-1])
    else:
        kwargs['dims'].append(1)

    return kwargs


def extract(kwargs: dict, layer: dict) -> dict:
    """
    Extracts a number of values from the tensor, returning two tensors

    Parameters
    ----------
    kwargs : dictionary
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
    layer : dictionary
        number : integer
            Number of values to extract from the previous layer

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(kwargs['dims'][-1] - layer['number'])
    kwargs['data_size'].append(kwargs['dims'][-1])
    return kwargs


def clone(kwargs: dict, _: dict) -> dict:
    """
    Constructs a layer to clone a number of values from the previous layer

    Parameters
    ----------
    kwargs : dictionary
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(kwargs['dims'][-1])
    kwargs['data_size'].append(kwargs['data_size'][-1])
    return kwargs


def index(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a layer to slice the output from the previous layer

    Parameters
    ----------
    kwargs : dictionary
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    layer : dictionary
        number : integer
            Index slice number;
        greater : boolean, default = True
            If slice should be values greater or less than number

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(kwargs['dims'][-1])

    if 'greater' in layer and (
            (layer['greater'] and layer['number'] < 0) or
            (not layer['greater'] and layer['number'] > 0)
    ):
        data_size = abs(layer['number'])
    else:
        data_size = kwargs['data_size'][-1] - abs(layer['number'])

    kwargs['data_size'].append(data_size)
    kwargs['module'].add_module(
        f"index_{kwargs['i']}",
        Index(layer['number'], greater=layer['greater']),
    )

    return kwargs


def concatenate(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a concatenation layer to combine the outputs from two layers

    Parameters
    ----------
    kwargs : dictionary
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
    layer : dictionary
        layer : integer
            Layer index to concatenate the previous layer output with
        channel : boolean
            If concatenation should be performed over the channels or last dimension

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    if 'channel' not in layer or layer['channel']:
        kwargs['dims'].append(kwargs['dims'][-1] + kwargs['dims'][layer['layer']])
        kwargs['data_size'].append(kwargs['data_size'][-1])
    else:
        kwargs['data_size'].append(kwargs['data_size'][-1] + kwargs['data_size'][layer['layer']])
        kwargs['dims'].append(kwargs['data_size'][-1])

    return kwargs


def shortcut(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a shortcut layer to add the outputs from two layers

    Parameters
    ----------
    kwargs : dictionary
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
    layer : dictionary
        layer : integer
            Layer index to add the previous layer output with;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    if kwargs['dims'][-1] == 1:
        kwargs['dims'].append(kwargs['dims'][layer['layer']])
    else:
        kwargs['dims'].append(kwargs['dims'][-1])

    kwargs['data_size'].append(kwargs['data_size'][-1])
    return kwargs


def skip(kwargs: dict, layer: dict) -> dict:
    """
    Bypasses previous layers by retrieving the output from the defined layer

    Parameters
    ----------
    kwargs : dictionary
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
    layer : dictionary
        layer : integer
            Layer index to retrieve the output;
    """
    kwargs['dims'].append(kwargs['dims'][layer['layer']])
    kwargs['data_size'].append(kwargs['data_size'][layer['layer']])
    return kwargs
