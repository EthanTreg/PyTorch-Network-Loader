"""
Convolutional network layers
"""
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


def convolutional(kwargs: dict, layer: dict) -> dict:
    """
    Convolutional layer constructor

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
        dropout_prob : float, optional
            Probability of dropout, not required if dropout from layer is False;
    layer : dictionary
        filters : integer
            Number of convolutional filters;
        2d : boolean, default = False
            If input data is 2D;
        dropout : boolean, default = True
            If dropout should be used;
        batch_norm : boolean, default = False
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;
        kernel : integer, default = 3
            Size of the kernel;
        stride : integer, default = 1
            Stride of the kernel;
        padding : integer | string, default = 'same'
            Input padding, can an integer or 'same' where 'same' preserves the input shape;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kernel_size = 3
    stride = 1
    padding = 'same'
    kwargs['dims'].append(layer['filters'])

    # Optional parameters
    if 'kernel' in layer:
        kernel_size = layer['kernel']

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
        in_channels=kwargs['dims'][-2],
        out_channels=kwargs['dims'][-1],
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode='replicate',
    )

    kwargs['module'].add_module(f"conv_{kwargs['i']}", conv)

    # Optional layers
    optional_layer(True, 'dropout', kwargs, layer, dropout(kwargs['dropout_prob']))
    optional_layer(False, 'batch_norm', kwargs, layer, batch_norm(kwargs['dims'][-1]))
    optional_layer(True, 'activation', kwargs, layer, nn.ELU())

    if padding != 'same':
        kwargs['data_size'].append(int(
            (kwargs['data_size'][-1] + 2 * padding - kernel_size) / stride + 1
        ))
    else:
        kwargs['data_size'].append(kwargs['data_size'][-1])

    return kwargs


def conv_upscale(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a 2x upscaler using a convolutional layer and pixel shuffling

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer], optional
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    layer : dictionary
        filters : integer
            Number of output convolutional filters;
        2d : boolean, default = False
            If input data is 2D;
        batch_norm : boolean, default = False
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;
        kernel : integer, default = 3
            Size of the kernel;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    layer['dropout'] = False
    layer['stride'] = 1
    layer['padding'] = 'same'
    layer['filters'] *= 2
    kwargs = convolutional(kwargs, layer)

    if ('2d' in layer and layer['2d']) or ('2d' not in layer and kwargs['2d']):
        pixel_shuffle = nn.PixelShuffle
    else:
        pixel_shuffle = PixelShuffle1d

    # Upscaling done using pixel shuffling
    kwargs['module'].add_module(f"pixel_shuffle_{kwargs['i']}", pixel_shuffle(2))
    kwargs['dims'][-1] = int(kwargs['dims'][-1] / 2)

    # Data size doubles
    kwargs['data_size'].append(kwargs['data_size'][-1] * 2)

    return kwargs


def conv_transpose(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a 2x upscaler using a transpose convolutional layer with fractional stride

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
        dropout_prob : float, optional
            Probability of dropout, not required if dropout from layer is False;
    layer : dictionary
        filters : integer
            Number of convolutional filters;
        2d : boolean, default = False
            If input data is 2D;
        dropout : boolean, default = True
            If dropout should be used;
        batch_norm : boolean, default = False
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(layer['filters'])

    if ('2d' in layer and layer['2d']) or ('2d' not in layer and kwargs['2d']):
        transpose = nn.ConvTranspose2d
        dropout = nn.Dropout2d
        batch_norm = nn.BatchNorm2d
    else:
        transpose = nn.ConvTranspose1d
        dropout = nn.Dropout1d
        batch_norm = nn.BatchNorm1d

    conv = transpose(
        in_channels=kwargs['dims'][-2],
        out_channels=kwargs['dims'][-1],
        kernel_size=2,
        stride=2,
    )

    kwargs['module'].add_module(f"conv_transpose_{kwargs['i']}", conv)

    # Optional layers
    optional_layer(True, 'dropout', kwargs, layer, dropout(kwargs['dropout_prob']))
    optional_layer(False, 'batch_norm', kwargs, layer, batch_norm(kwargs['dims'][-1]))
    optional_layer(True, 'activation', kwargs, layer, nn.ELU())

    # Data size doubles
    kwargs['data_size'].append(kwargs['data_size'][-1] * 2)

    return kwargs


def conv_depth_downscale(kwargs: dict, layer: dict) -> dict:
    """
    Constructs depth downscaler using convolution with kernel size of 1

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    layer : dictionary
        2d : boolean, default = False
            If input data is 2D;
        batch_norm : boolean, default = False
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;

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
            Layer number;
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
        dropout_prob : float, optional
            Probability of dropout, not required if dropout from layer is False;
    layer : dictionary
        filters : integer
            Number of convolutional filters;
        2d : boolean, default = False
            If input data is 2D;
        dropout : boolean, default = True
            If dropout should be used;
        batch_norm : boolean, default = False
            If batch normalisation should be used;
        activation : boolean, default = True
            If ELU activation should be used;

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


def pool(kwargs: dict, layer: dict) -> dict:
    """
    Constructs a max pooling layer for 2x downscaling

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
        2d : boolean, default = False
            If input data is 2D

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(int(kwargs['dims'][-1]))

    if ('2d' in layer and layer['2d']) or ('2d' not in layer and kwargs['2d']):
        max_pool = nn.MaxPool2d
    else:
        max_pool = nn.MaxPool1d

    kwargs['module'].add_module(f"pool_{kwargs['i']}", max_pool(kernel_size=2))

    # Data size halves
    kwargs['data_size'].append(int(kwargs['data_size'][-1] / 2))

    return kwargs
