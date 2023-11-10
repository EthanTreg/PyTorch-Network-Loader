"""
Convolutional network layers
"""
import numpy as np
from torch import nn, Tensor

from netloader.layers.utils import optional_layer


class PixelShuffle1d(nn.Module):
    """
    Used for upscaling by scale factor r for an input (*, C x r, L) to an output (*, C, L x r)

    Equivalent to torch.nn.PixelShuffle but for 1D

    Attributes
    ----------
    upscale_factor : integer
        Upscaling factor

    Methods
    -------
    forward(x)
        Forward pass of PixelShuffle1D
    """
    def __init__(self, upscale_factor: int):
        """
        Parameters
        ----------
        upscale_factor : integer
            Upscaling factor
        """
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of pixel shuffle

        Parameters
        ----------
        x : Tensor, shape (*, C x r, L)
            Input tensor

        Returns
        -------
        Tensor, (*, C, L x r)
            Output tensor
        """
        output_channels = x.size(1) // self.upscale_factor
        output_size = self.upscale_factor * x.size(2)

        x = x.view([x.size(0), self.upscale_factor, output_channels, x.size(2)])
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.size(0), output_channels, output_size)
        return x


def _kernel_shape(stride: int, kernel: int | list[int], padding: int | list[int], shape: list[int]):
    """
    Calculates the output shape after a kernel operation

    Parameters
    ----------
    kernel : integer
        Size of the kernel
    stride : integer
        Stride of the kernel
    padding : integer | list[integer]
        Input padding
    shape : list[integer]
        Input shape into the layer
    """
    if isinstance(kernel, int):
        kernel = [kernel] * len(shape[1:])

    if isinstance(padding, int):
        padding = [padding] * len(shape[1:])

    for i, (kernel_length, pad, length) in enumerate(zip(kernel, padding, shape[1:])):
        shape[i + 1] = int((length + 2 * pad - kernel_length) / stride + 1)

    return shape


def _same_padding(
        kernel: int,
        stride: int,
        in_shape: list[int],
        out_shape: list[int]) -> list[int]:
    """
    Calculates the padding required for specific output shape

    Parameters
    ----------
    kernel : integer
        Size of the kernel
    stride : integer
        Stride of the kernel
    in_shape : list[integer]
        Input shape into the layer
    out_shape : list[integer]
        Output shape from the layer

    Returns
    -------
    list[integer]
        Required padding for specific output shape
    """
    padding = []

    for in_length, out_length in zip(in_shape[1:], out_shape[1:]):
        padding.append(int((stride * (out_length - 1) + kernel - in_length) / 2))

    return padding


def convolutional(kwargs: dict, layer: dict) -> dict:
    """
    Convolutional layer constructor

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
        module : Sequential
            Sequential module to contain the layer
        out_shape : list[integer], optional
            Shape of the network's output, required only if layer contains factor
        dropout_prob : float, optional
            Probability of dropout, not required if dropout from layer is False
    layer : dictionary
        filters : integer, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        2d : boolean, default = False
            If input data is 2D
        dropout : boolean, default = True
            If dropout should be used
        batch_norm : boolean, default = False
            If batch normalisation should be used
        activation : boolean, default = True
            If ELU activation should be used
        kernel : integer | list[integer], default = 3
            Size of the kernel
        stride : integer, default = 1
            Stride of the kernel
        padding : integer | string | list[integer], default = 'same'
            Input padding, can an integer, list of integers or
            'same' where 'same' preserves the input shape

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kernel = 3
    stride = 1
    padding = 'same'
    shape = kwargs['shape'][-1].copy()

    if 'factor' not in layer:
        shape[0] = layer['filters']
    else:
        shape[0] = max(1, int(kwargs['out_shape'][0] * layer['factor']))

    # Optional parameters
    if 'kernel' in layer:
        kernel = layer['kernel']

    if 'stride' in layer:
        stride = layer['stride']

    if 'padding' in layer:
        padding = layer['padding']

    if ('2d' in layer and layer['2d']) or ('2d' not in layer and kwargs['2d']):
        conv = nn.Conv2d
        dropout = nn.Dropout2d
        batch_norm = nn.BatchNorm2d
    else:
        conv = nn.Conv1d
        dropout = nn.Dropout1d
        batch_norm = nn.BatchNorm1d

    conv = conv(
        in_channels=kwargs['shape'][-1][0],
        out_channels=shape[0],
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        padding_mode='replicate',
    )

    kwargs['module'].add_module(f"conv_{kwargs['i']}", conv)

    # Optional layers
    optional_layer(True, 'dropout', kwargs, layer, dropout(kwargs['dropout_prob']))
    optional_layer(False, 'batch_norm', kwargs, layer, batch_norm(shape[0]))
    optional_layer(True, 'activation', kwargs, layer, nn.ELU())

    if padding != 'same':
        kwargs['shape'].append(_kernel_shape(stride, kernel, padding, shape))
    else:
        kwargs['shape'].append(shape)

    return kwargs


def conv_depth_downscale(kwargs: dict, layer: dict) -> dict:
    """
    Constructs depth downscaler using convolution with kernel size of 1

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
        2d : boolean, default = False
            If input data is 2D
        batch_norm : boolean, default = False
            If batch normalisation should be used
        activation : boolean, default = True
            If ELU activation should be used

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    layer['dropout'] = False
    layer['filters'] = 1
    layer['kernel'] = 1
    layer['stride'] = 1
    layer['padding'] = 'same'

    convolutional(kwargs, layer)
    return kwargs


def conv_downscale(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a convolutional layer with stride 2 for 2x downscaling

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
        module : Sequential
            Sequential module to contain the layer
        out_shape : list[integer], optional
            Shape of the network's output, required only if layer contains factor
        dropout_prob : float, optional
            Probability of dropout, not required if dropout from layer is False
    layer : dictionary
        filters : integer, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        2d : boolean, default = False
            If input data is 2D
        dropout : boolean, default = True
            If dropout should be used
        batch_norm : boolean, default = False
            If batch normalisation should be used
        activation : boolean, default = True
            If ELU activation should be used

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    layer['kernel'] = 3
    layer['stride'] = 2
    layer['padding'] = 1

    convolutional(kwargs, layer)
    return kwargs


def conv_transpose(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a 2x upscaler using a transpose convolutional layer with fractional stride

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
        module : Sequential
            Sequential module to contain the layer
        out_shape : list[integer], optional
            Shape of the network's output, required only if layer contains factor
        dropout_prob : float, optional
            Probability of dropout, not required if dropout from layer is False
    layer : dictionary
        filters : integer, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        2d : boolean, default = False
            If input data is 2D
        dropout : boolean, default = True
            If dropout should be used
        batch_norm : boolean, default = False
            If batch normalisation should be used
        activation : boolean, default = True
            If ELU activation should be used

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['shape'].append(kwargs['shape'][-1].copy())

    if 'factor' not in layer:
        kwargs['shape'][-1][0] = layer['filters']
    else:
        kwargs['shape'][-1][0] = max(1, int(kwargs['out_shape'][0] * layer['factor']))

    if ('2d' in layer and layer['2d']) or ('2d' not in layer and kwargs['2d']):
        transpose = nn.ConvTranspose2d
        dropout = nn.Dropout2d
        batch_norm = nn.BatchNorm2d
    else:
        transpose = nn.ConvTranspose1d
        dropout = nn.Dropout1d
        batch_norm = nn.BatchNorm1d

    conv = transpose(
        in_channels=kwargs['shape'][-2][0],
        out_channels=kwargs['shape'][-1][0],
        kernel_size=2,
        stride=2,
    )

    kwargs['module'].add_module(f"conv_transpose_{kwargs['i']}", conv)

    # Optional layers
    optional_layer(True, 'dropout', kwargs, layer, dropout(kwargs['dropout_prob']))
    optional_layer(False, 'batch_norm', kwargs, layer, batch_norm(kwargs['shape'][-1][0]))
    optional_layer(True, 'activation', kwargs, layer, nn.ELU())

    # Data size doubles
    kwargs['shape'][-1][1:] = [length * 2 for length in kwargs['shape'][-1][1:]]
    return kwargs


def conv_upscale(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a 2x upscaler using a convolutional layer and pixel shuffling

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
        out_shape : list[integer], optional
            Shape of the network's output, required only if layer contains factor
        module : Sequential
            Sequential module to contain the layer
    layer : dictionary
        filters : integer, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        2d : boolean, default = False
            If input data is 2D
        batch_norm : boolean, default = False
            If batch normalisation should be used
        activation : boolean, default = True
            If ELU activation should be used
        kernel : integer | list[integer], default = 3
            Size of the kernel

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    layer['dropout'] = False
    layer['stride'] = 1
    layer['padding'] = 'same'

    if 'factor' in layer:
        layer['factor'] *= 2
    else:
        layer['filters'] *= 2

    kwargs = convolutional(kwargs, layer)

    if ('2d' in layer and layer['2d']) or ('2d' not in layer and kwargs['2d']):
        pixel_shuffle = nn.PixelShuffle
    else:
        pixel_shuffle = PixelShuffle1d

    # Upscaling done using pixel shuffling
    kwargs['module'].add_module(f"pixel_shuffle_{kwargs['i']}", pixel_shuffle(2))
    kwargs['shape'][-1][0] = int(kwargs['shape'][-1][0] / 2)

    # Data size doubles
    kwargs['shape'][-1][1:] = [length * 2 for length in kwargs['shape'][-1][1:]]
    return kwargs


def pool(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a max pooling layer

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
        2d : boolean, default = False
            If input data is 2D
        kernel : integer, default = 2
            Size of the kernel
        stride : integer, default = 2
            Stride of the kernel
        padding : integer | string, default = 0
            Input padding, can an integer or 'same' where 'same' preserves the input shape
        mode : string, default = 'max'
            Whether to use max pooling (max) or average pooling (average)

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kernel = stride = 2
    padding = 0
    mode = np.array([[nn.MaxPool1d, nn.MaxPool2d], [nn.AvgPool1d, nn.AvgPool2d]])

    # Optional parameters
    if 'kernel' in layer:
        kernel = layer['kernel']

    if 'stride' in layer:
        stride = layer['stride']

    if 'padding' in layer:
        padding = layer['padding']

    if padding == 'same':
        padding = _same_padding(kernel, stride, kwargs['shape'][-1], kwargs['shape'][-1])

    if ('2d' in layer and layer['2d']) or ('2d' not in layer and kwargs['2d']):
        mode = mode[:, 1]
    else:
        mode = mode[:, 0]

    if 'mode' not in layer or layer['mode'] != 'average':
        mode = mode[0]
    else:
        mode = mode[1]

    kwargs['module'].add_module(
        f"pool_{kwargs['i']}",
        mode(kernel_size=kernel, stride=stride, padding=padding),
    )

    if padding != 'same':
        kwargs['shape'].append(_kernel_shape(stride, kernel, padding, kwargs['shape'][-1]))
    else:
        kwargs['shape'].append(kwargs['shape'][-1].copy())
    return kwargs


def pool_downscale(kwargs: dict, layer: dict) -> dict:
    """
    Downscales the input using max pooling
    
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
        factor : integer
            Factor of downscaling
        2d : boolean, default = False
            If input data is 2D
        mode : string, default = 'max'
            Whether to use max pooling (max) or average pooling (average)
    """
    layer['kernel'] = layer['stride'] = layer['factor']
    layer['padding'] = 0
    return pool(kwargs, layer)
