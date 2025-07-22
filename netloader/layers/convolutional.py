"""
Convolutional network layers
"""
from typing import Any, Type

import torch
import numpy as np
from torch import nn, Tensor

from netloader.layers.misc import LayerNorm
from netloader.layers.utils import _int_list_conversion, _kernel_shape, _padding
from netloader.layers.base import BaseLayer


class Conv(BaseLayer):
    """
    Convolutional layer constructor

    Supports 1D, 2D, and 3D convolution

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            net_out: list[int],
            shapes: list[list[int]],
            filters: int | None = None,
            layer: int | None = None,
            factor: float | None = None,
            groups: int = 1,
            kernel: int | list[int] = 3,
            stride: int | list[int] = 1,
            padding: int | str | list[int] = 0,
            dropout: float = 0,
            activation: str | None = 'ELU',
            norm: str | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_out : list[int]
            Shape of the network's output, required only if layer contains factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        layer : int, optional
            If factor is not None, which layer for factor to be relative to, if None, network output
            will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels, or if layer is provided,
            the layer's channels, multiplied by factor, won't be used if filters is provided
        groups : int, default = 1
            Number of input channel groups, each with its own convolutional filter(s), input and
            output channels must both be divisible by the number of groups
        kernel : int | list[int], default = 3
            Size of the kernel
        stride : int | list[int], default = 1
            Stride of the kernel
        padding : int | str | list[int], default = 0
            Input padding, can an int, list of ints or 'same' where 'same' preserves the input shape
        dropout : float, default = 0
            Probability of dropout
        activation : str | None, default = 'ELU'
            Which activation function to use from PyTorch
        norm : {None, 'batch', 'layer'}, default = None
            If batch or layer normalisation should be used

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        padding_: int | str | list[int] = padding
        shape: list[int] = shapes[-1].copy()
        target: list[int] = shapes[layer] if layer is not None else net_out
        conv: Type[nn.Module]
        dropout_: Type[nn.Module]
        batch_norm_: Type[nn.Module]

        # Check for errors and calculate same padding
        if isinstance(padding, str):
            self._check_options('padding', padding, {'same'})
            self._check_stride(stride)
            padding = _padding(kernel, stride, shapes[-1], shapes[-1])

        # Check for errors and calculate number of filters
        self._check_shape(shape)
        self._check_factor_filters(shape, filters, factor, target)
        self._check_groups(shapes[-1][0], shape[0], groups)
        self._check_options('norm', norm, {None, 'batch', 'layer'})

        conv, dropout_, batch_norm_ = [
            [nn.Conv1d, nn.Dropout1d, nn.BatchNorm1d],
            [nn.Conv2d, nn.Dropout2d, nn.BatchNorm2d],
            [nn.Conv3d, nn.Dropout3d, nn.BatchNorm3d],
        ][len(shape) - 2]

        self.layers.add_module('Conv', conv(
            in_channels=shapes[-1][0],
            out_channels=shape[0],
            kernel_size=kernel,
            stride=stride,
            padding=padding_,
            groups=groups,
            padding_mode='replicate',
        ))

        # Optional layers
        if activation:
            self.layers.add_module('Activation', getattr(nn, activation)())

        if norm == 'batch':
            self.layers.add_module('BatchNorm', batch_norm_(shape[0]))
        elif norm == 'layer':
            self.layers.add_module('LayerNorm', LayerNorm(shape=shape[0:1]))

        if dropout:
            self.layers.add_module('Dropout', dropout_(dropout))

        if padding_ == 'same':
            shapes.append(shape)
        else:
            assert not isinstance(padding, str)
            shapes.append(_kernel_shape(kernel, stride, padding, shape))

    @staticmethod
    def _check_groups(in_channels: int, out_channels: int, groups: int) -> None:
        """
        Checks if the number of groups is compatible with the number of input and output channels

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        groups : int
            Number of groups
        """
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(f'Number of groups ({groups}) is not compatible with input channels '
                             f'({in_channels}) and/or output channels ({out_channels}), check that '
                             f'they are divisible by groups')


class ConvDepth(Conv):
    """
    Constructs a depthwise convolutional layer

    Supports 1D, 2D, and 3D convolution

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            net_out: list[int],
            shapes: list[list[int]],
            filters: int | None = None,
            layer: int | None = None,
            factor: float | None = None,
            kernel: int | list[int] = 3,
            stride: int | list[int] = 1,
            padding: int | str | list[int] = 0,
            dropout: float = 0,
            activation: str | None = 'ELU',
            norm: str | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_out : list[int]
            Shape of the network's output, required only if layer contains factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        layer : int, optional
            If factor is not None, which layer for factor to be relative to, if None, network output
            will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        kernel : int | list[int], default = 3
            Size of the kernel
        stride : int | list[int], default = 1
            Stride of the kernel
        padding : int | str | list[int], default = 0
            Input padding, can an int, list of ints or 'same' where 'same' preserves the input shape
        dropout : float, default = 0
            Probability of dropout
        activation : str | None, default = 'ELU'
            Which activation function to use
        norm : {None, 'batch', 'layer'}, default = None
            If batch or layer normalisation should be used

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            net_out=net_out,
            shapes=shapes,
            filters=filters,
            layer=layer,
            factor=factor,
            groups=shapes[-1][0],
            kernel=kernel,
            stride=stride,
            padding=padding,
            dropout=dropout,
            activation=activation,
            norm=norm,
            **kwargs,
        )


class ConvDepthDownscale(Conv):
    """
    Constructs depth downscaler using convolution with kernel size of 1

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            net_out: list[int],
            shapes: list[list[int]],
            dropout: float = 0,
            activation: str | None = 'ELU',
            norm: str | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_out : list[int]
            Shape of the network's output, required only if layer contains factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        dropout : float, default = 0
            Probability of dropout
        activation : str | None, default = 'ELU'
            Which activation function to use
        norm : {None, 'batch', 'layer'}, default = None
            If batch or layer normalisation should be used

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            net_out=net_out,
            shapes=shapes,
            filters=1,
            stride=1,
            kernel=1,
            padding='same',
            dropout=dropout,
            activation=activation,
            norm=norm,
            **kwargs,
        )


class ConvDownscale(Conv):
    """
    Constructs a strided convolutional layer for downscaling

    The scale factor is equal to the stride and kernel size

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass
    """
    def __init__(self,
                 net_out: list[int],
                 shapes: list[list[int]],
                 filters: int | None = None,
                 layer: int | None = None,
                 factor: float | None = None,
                 scale: int = 2,
                 dropout: float = 0,
                 activation: str | None = 'ELU',
                 norm: str | None = None,
                 **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_out : list[int]
            Shape of the network's output, required only if layer contains factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        layer : int, optional
            If factor is not None, which layer for factor to be relative to, if None, network output
            will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        scale : int, default = 2
            Stride and size of the kernel, which acts as the downscaling factor
        dropout : float, default = 0
            Probability of dropout
        activation : str | None, default = 'ELU'
            Which activation function to use
        norm : {None, 'batch', 'layer'}, default = None
            If batch or layer normalisation should be used

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            net_out=net_out,
            shapes=shapes,
            filters=filters,
            layer=layer,
            factor=factor,
            kernel=scale,
            stride=scale,
            padding=0,
            dropout=dropout,
            activation=activation,
            norm=norm,
            **kwargs,
        )


class ConvTranspose(BaseLayer):
    """
    Constructs a transpose convolutional layer with fractional stride for input upscaling

    Supports 1D, 2D, and 3D transposed convolution

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the transposed convolutional layer
    """
    def __init__(
            self,
            net_out: list[int],
            shapes: list[list[int]],
            filters: int | None = None,
            layer: int | None = None,
            factor: float | None = None,
            kernel: int | list[int] = 3,
            stride: int | list[int] = 1,
            out_padding: int | list[int] = 0,
            dilation: int | list[int] = 1,
            padding: int | str | list[int] = 0,
            dropout: float = 0,
            activation: str | None = 'ELU',
            norm: str | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_out : list[int]
            Shape of the network's output, required only if layer contains factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        layer : int, optional
            If factor is not None, which layer for factor to be relative to, if None, network output
            will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        kernel : int | list[int], default = 3
            Size of the kernel
        stride : int | list[int], default = 1
            Stride of the kernel
        out_padding : int | list[int], default = 0
            Padding applied to the output
        dilation : int | list[int], default = 1
            Spacing between kernel points
        padding : int | str | list[int], default = 0
            Inverse of convolutional padding which removes rows from each dimension in the output
        dropout : float, default =  0
            Probability of dropout
        activation : str | None, default = 'ELU'
            Which activation function to use
        norm : {None, 'batch', 'layer'}, default = None
            If batch or layer normalisation should be used

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._slice: np.ndarray = np.array([slice(None)] * len(shapes[-1][1:]))
        padding_: int | str | list[int] = padding
        shape: list[int] = shapes[-1].copy()
        target: list[int] = shapes[layer] if layer is not None else net_out
        transpose: Type[nn.Module]
        dropout_: Type[nn.Module]
        batch_norm_: Type[nn.Module]

        # Check for errors and calculate same padding
        if isinstance(padding, str):
            self._check_options('padding', padding, {'same'})
            padding = _padding_transpose(kernel, stride, dilation, shapes[-1], shape)

        # Check for errors and calculate number of filters
        self._check_shape(shape)
        self._check_out_padding(stride, dilation, out_padding)
        self._check_factor_filters(shape, filters, factor, target)
        self._check_options('norm', norm, {None, 'batch', 'layer'})

        transpose, dropout_, batch_norm_ = [
            [nn.ConvTranspose1d, nn.Dropout1d, nn.BatchNorm1d],
            [nn.ConvTranspose2d, nn.Dropout2d, nn.BatchNorm2d],
            [nn.ConvTranspose3d, nn.Dropout3d, nn.BatchNorm3d],
        ][len(shape) - 2]

        assert not isinstance(padding, str)
        shape = _kernel_transpose_shape(
            kernel,
            stride,
            padding,
            dilation,
            out_padding,
            shape,
        )

        # Correct same padding for one-sided padding
        if padding_ == 'same' and shape != shapes[-1]:
            self._slice[np.array(shape[1:]) - np.array(shapes[-1][1:]) == 1] = slice(-1)
            shape[1:] = shapes[-1][1:]

        self.layers.add_module('Transpose', transpose(
            in_channels=shapes[-1][0],
            out_channels=shape[0],
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            output_padding=out_padding,
            dilation=dilation,
        ))

        # Optional layers
        if activation:
            self.layers.add_module('Activation', getattr(nn, activation)())

        if norm == 'batch':
            self.layers.add_module('BatchNorm', batch_norm_(shape[0]))
        elif norm == 'layer':
            self.layers.add_module('LayerNorm', LayerNorm(shape=shape[0:1]))

        if dropout:
            self.layers.add_module('Dropout', dropout_(dropout))

        shapes.append(shape)

    @staticmethod
    def _check_out_padding(
            stride: int | list[int],
            dilation: int | list[int],
            out_padding: int | list[int]) -> None:
        """
        Checks if the output padding is compatible with the dilation and stride

        Parameters
        ----------
        stride : int | list[int]
            Stride of the kernel
        dilation : int | list[int]
            Dilation of the kernel
        out_padding : int | list[int]
            Output padding
        """
        if ((np.array(out_padding) >= np.array(stride)) *
                (np.array(out_padding) >= np.array(dilation))).any():
            raise ValueError(f'Output padding ({out_padding}) must be smaller than either stride '
                             f'({stride}) or dilation ({dilation})')

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward pass of the transposed convolutional layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor with batch size N

        *args
            Optional arguments to pass to the parent forward method
        **kwargs
            Optional keyword arguments to pass to the parent forward method

        Returns
        -------
        (N,...) Tensor
            Output tensor with batch size N
        """
        x = super().forward(x, *args, **kwargs)
        return x[..., *self._slice]


class ConvTransposeUpscale(ConvTranspose):
    """
    Constructs an upscaler using a transposed convolutional layer.

    Supports 1D, 2D, and 3D transposed convolutional upscaling.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            net_out: list[int],
            shapes: list[list[int]],
            filters: int | None = None,
            layer: int | None = None,
            factor: float | None = None,
            scale: int | list[int] = 2,
            out_padding: int | list[int] = 0,
            dropout: float = 0,
            activation: str | None = 'ELU',
            norm: str | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_out : list[int]
            Shape of the network's output, required only if layer contains factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        layer : int, optional
            If factor is not None, which layer for factor to be relative to, if None, network output
            will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        scale : int | list[int], default = 2
            Stride and size of the kernel, which acts as the upscaling factor
        out_padding : int | list[int], default = 0
            Padding applied to the output
        dropout : float, default =  0
            Probability of dropout
        activation : str | None, default = 'ELU'
            Which activation function to use
        norm : {None, 'batch', 'layer'}, default = None
            If batch or layer normalisation should be used

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            net_out=net_out,
            shapes=shapes,
            filters=filters,
            layer=layer,
            factor=factor,
            kernel=scale,
            stride=scale,
            padding=0,
            out_padding=out_padding,
            dropout=dropout,
            activation=activation,
            norm=norm,
            **kwargs,
        )


class ConvUpscale(Conv):
    """
    Constructs an upscaler using a convolutional layer and pixel shuffling.

    Supports 1D, 2D, and 3D convolutional upscaling.

    See `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel
    Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_ by Shi et al. (2016) for
    details.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            net_out: list[int],
            shapes: list[list[int]],
            filters: int | None = None,
            layer: int | None = None,
            factor: float | None = None,
            scale: int = 2,
            kernel: int | list[int] = 3,
            dropout: float = 0,
            activation: str | None = 'ELU',
            norm: str | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_out : list[int]
            Shape of the network's output, required only if layer contains factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        layer : int, optional
            If factor is not None, which layer for factor to be relative to, if None, network output
            will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        scale : int, default = 2
            Factor to upscale the input by
        kernel : int | list[int], default = 3
            Size of the kernel
        dropout : float, default =  0
            Probability of dropout
        activation : str | None, default = 'ELU'
            Which activation function to use
        norm : {None, 'batch', 'layer'}, default = None
            If batch or layer normalisation should be used

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        filters_scale: int = scale ** (len(shapes[-1]) - 1)
        filters = self._check_factor_filters(
            shapes[-1].copy(),
            filters,
            factor,
            shapes[layer] if layer is not None else net_out,
        )[0] * filters_scale

        # Convolutional layer
        super().__init__(
            net_out=net_out,
            shapes=shapes,
            filters=filters,
            kernel=kernel,
            stride=1,
            padding='same',
            dropout=dropout,
            activation=activation,
            norm=norm,
            **kwargs,
        )

        # Upscaling done using pixel shuffling
        self.layers.add_module('PixelShuffle', PixelShuffle(scale))
        shapes[-1][0] = shapes[-1][0] // filters_scale
        shapes[-1][1:] = [length * scale for length in shapes[-1][1:]]


class PixelShuffle(BaseLayer):
    r"""
    Used for upscaling by scale factor :math:`r` for an input :math:`(N,C\times r^n,D_1,...,D_n)` to
    an output :math:`(N,C,D_1\times r,...,D_n\times r)`

    Equivalent to :class:`torch.nn.PixelShuffle`, but for nD

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of pixel shuffle
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(self, scale: int, shapes: list[list[int]] | None = None, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        scale : int
            Upscaling factor
        shapes : list[list[int]], default = None
            Shape of the outputs from each layer

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**({'idx': 0} | kwargs))
        self._scale: int = scale
        filters_scale: int

        # If not used as an individual layer in Network
        if not shapes:
            return

        filters_scale = self._scale ** (len(shapes[-1][1:]))
        self._check_filters(filters_scale, shapes[-1])

        shapes.append(shapes[-1].copy())
        shapes[-1][0] = shapes[-1][0] // filters_scale
        shapes[-1][1:] = [length * self._scale for length in shapes[-1][1:]]

    @staticmethod
    def _check_filters(filters_scale: int, shape: list[int]) -> None:
        """
        Checks if the number of channels is an integer multiple of the upscaling factor

        Parameters
        ----------
        filters_scale : int
            Upscaling factor for the number of filters
        shape : list[int]
            Shape of the input
        """
        if shape[0] % filters_scale != 0:
            raise ValueError(f'Channels ({shape}) must be an integer multiple of '
                             f'{filters_scale}')

    def forward(self, x: Tensor, *_: Any, **__: Any) -> Tensor:
        r"""
        Forward pass of pixel shuffle

        Parameters
        ----------
        x : :math:`(N,C\times r^n,D_1,...,D_n)` Tensor
            Input tensor

        Returns
        -------
        :math:`(N,C,D_1\times r,...,D_n\times r)` Tensor
            Output tensor
        """
        dims: int
        filters_scale: int = self._scale ** (len(x.shape[2:]))
        output_channels: int = x.size(1) // filters_scale
        output_shape: Tensor = self._scale * torch.tensor(x.shape[2:])
        idxs: Tensor

        dims = len(output_shape)
        idxs = torch.arange(dims * 2) + 2

        x = x.view([x.size(0), output_channels, *[self._scale] * len(x.shape[2:]), *x.shape[2:]])
        x = x.permute(0, 1, *torch.ravel(torch.column_stack((idxs[dims:], idxs[:dims]))))
        x = x.reshape(x.size(0), output_channels, *output_shape)
        return x

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'upscale_factor={self._scale}'


def _kernel_transpose_shape(
        kernel: int | list[int],
        strides: int | list[int],
        padding: int | list[int],
        dilation: int | list[int],
        out_padding: int | list[int],
        shape: list[int]) -> list[int]:
    """
    Calculates the output shape after a transposed convolutional kernel

    Parameters
    ----------
    kernel : int | list[int]
        Size of the kernel
    strides : int | list[int]
        Stride of the kernel
    padding : int | list[int]
        Input padding
    dilation : int | list[int]
        Spacing between kernel elements
    shape : list[int]
        Input shape of the layer

    Returns
    -------
    list[int]
        Output shape of the layer
    """
    shape = shape.copy()
    strides, kernel, padding, dilation, out_padding = _int_list_conversion(
        len(shape[1:]),
        [strides, kernel, padding, dilation, out_padding]
    )

    for i, (stride, kernel_length, pad, dilation_length, out_pad, length) in enumerate(zip(
            strides,
            kernel,
            padding,
            dilation,
            out_padding,
            shape[1:]
    )):
        shape[i + 1] = max(
            1,
            stride * (length - 1) + dilation_length * (kernel_length - 1) - 2 * pad + out_pad + 1
        )

    return shape


def _padding_transpose(
        kernel: int | list[int],
        strides: int | list[int],
        dilation: int | list[int],
        in_shape: list[int],
        out_shape: list[int]) -> list[int]:
    """
    Calculates the padding required for specific output shape

    Parameters
    ----------
    kernel : int | list[int]
        Size of the kernel
    strides : int | list[int]
        Stride of the kernel
    dilation : int | list[int]
        Spacing between kernel elements
    in_shape : list[int]
        Input shape of the layer
    out_shape : list[int]
        Output shape of the layer

    Returns
    -------
    list[int]
        Required padding for specific output shape
    """
    padding: list[int] = []
    strides, kernel, dilation = _int_list_conversion(
        len(in_shape[1:]),
        [strides, kernel, dilation],
    )

    for stride, kernel_length, dilation_length, in_length, out_length in zip(
            strides,
            kernel,
            dilation,
            in_shape[1:],
            out_shape[1:],
    ):
        padding.append(
            (stride * (in_length - 1) + dilation_length * (kernel_length - 1) - out_length + 1) // 2
        )

    return padding
