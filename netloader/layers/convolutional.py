"""
Convolutional network layers
"""
from typing import Any

import torch
import numpy as np
from torch import nn, Tensor

from netloader.layers.utils import BaseLayer, _int_list_conversion, _kernel_shape


class Conv(BaseLayer):
    """
    Convolutional layer constructor

    Supports 1D, 2D, and 3D convolution

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            idx: int,
            shapes: list[list[int]],
            filters: int | None = None,
            factor: float | None = None,
            net_out: list[int] | None = None,
            batch_norm: bool = False,
            activation: bool = True,
            kernel: int | list[int] = 3,
            stride: int | list[int] = 1,
            padding: int | str | list[int] = 'same',
            dropout: float = 0.1,
            **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        net_out : list[int], optional
            Shape of the network's output, required only if layer contains factor
        batch_norm : bool, default = False
            If batch normalisation should be used
        activation : bool, default = True
            If ELU activation should be used
        kernel : int | list[int], default = 3
            Size of the kernel
        stride : int | list[int], default = 1
            Stride of the kernel
        padding : int | str | list[int], default = 'same'
            Input padding, can an int, list of ints or
            'same' where 'same' preserves the input shape
        dropout : float, default = 0.1
            Probability of dropout


        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        self._activation: bool = activation
        self._batch_norm: bool = batch_norm
        self._dropout: float = dropout
        shape: list[int] = shapes[-1].copy()
        conv: nn.Module
        dropout_: nn.Module
        batch_norm_: nn.Module

        if isinstance(padding, str) and padding != 'same':
            raise ValueError(f'Convolution padding of {padding} in layer {idx}, padding must be '
                             f"either 'same', int, or list[int]")

        if padding != 'same' and (np.array(shape[1:]) + np.array(padding) < np.array(kernel)).any():
            raise ValueError(f'Convolution kernel of {kernel} in layer {idx} is too large for '
                             f'input shape {shape} and padding {padding}')

        if (np.array(stride) > 1).any() and padding == 'same':
            raise ValueError(f"Convolution 'same' padding in layer {idx} is not supported for "
                             f'strides > 1, stride is {stride}')

        if not 1 < len(shape) < 5:
            raise ValueError(f'Convolution in layer {idx} does not support tensors with more than '
                             f'4 dimensions or less than 2, input shape is {shape}')

        if factor is not None and net_out is not None:
            shape[0] = max(1, int(net_out[0] * factor))
        elif filters:
            shape[0] = filters
        else:
            raise ValueError(f'Convolution in layer {idx} requires either factor or filters')

        conv, dropout_, batch_norm_ = [
            [nn.Conv1d, nn.Dropout1d, nn.BatchNorm1d],
            [nn.Conv2d, nn.Dropout2d, nn.BatchNorm2d],
            [nn.Conv3d, nn.Dropout3d, nn.BatchNorm3d],
        ][len(shape) - 2]

        self.layers.append(conv(
            in_channels=shapes[-1][0],
            out_channels=shape[0],
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            padding_mode='replicate',
        ))

        # Optional layers
        if self._activation:
            self.layers.append(nn.ELU())

        if self._batch_norm:
            self.layers.append(batch_norm_(shape[0]))

        if self._dropout:
            self.layers.append(dropout_(self._dropout))

        if padding == 'same':
            shapes.append(shape)
        else:
            assert not isinstance(padding, str)
            shapes.append(_kernel_shape(kernel, stride, padding, shape))


class ConvDepthDownscale(Conv):
    """
    Constructs depth downscaler using convolution with kernel size of 1

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            idx: int,
            shapes: list[list[int]],
            net_out: list[int],
            batch_norm: bool = False,
            activation: bool = True,
            dropout: float = 0,
            **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        net_out : list[int], optional
            Shape of the network's output, required only if layer contains factor
        batch_norm : bool, default = False
            If batch normalisation should be used
        activation : bool, default = True
            If ELU activation should be used
        dropout : float, default = 0
            Probability of dropout

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            idx=idx,
            shapes=shapes,
            filters=1,
            net_out=net_out,
            batch_norm=batch_norm,
            activation=activation,
            stride=1,
            dropout=dropout,
            kernel=1,
            padding='same',
            **kwargs,
        )


class ConvDownscale(Conv):
    """
    Constructs a strided convolutional layer for downscaling

    The scale factor is equal to the stride and kernel size

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass
    """
    def __init__(self,
                 idx: int,
                 shapes: list[list[int]],
                 filters: int | None = None,
                 factor: float | None = None,
                 net_out: list[int] | None = None,
                 batch_norm: bool = False,
                 activation: bool = True,
                 scale: int = 2,
                 dropout: float = 0.1,
                 **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        net_out : list[int], optional
            Shape of the network's output, required only if layer contains factor
        batch_norm : bool, default = False
            If batch normalisation should be used
        activation : bool, default = True
            If ELU activation should be used
        scale : int, default = 2
            Stride and size of the kernel, which acts as the downscaling factor
        dropout : float, default = 0.1
            Probability of dropout

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            idx=idx,
            shapes=shapes,
            filters=filters,
            factor=factor,
            net_out=net_out,
            batch_norm=batch_norm,
            activation=activation,
            stride=scale,
            dropout=dropout,
            kernel=scale,
            padding=0,
            **kwargs,
        )


class ConvTranspose(BaseLayer):
    """
    Constructs a transpose convolutional layer with fractional stride for input upscaling

    Supports 1D, 2D, and 3D transposed convolution

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the transposed convolutional layer
    """
    def __init__(
            self,
            idx: int,
            shapes: list[list[int]],
            filters: int | None = None,
            factor: float | None = None,
            net_out: list[int] | None = None,
            batch_norm: bool = False,
            activation: bool = True,
            kernel: int | list[int] = 3,
            stride: int | list[int] = 1,
            out_padding: int | list[int] = 0,
            dilation: int | list[int] = 1,
            padding: int | str | list[int] = 'same',
            dropout: float = 0.1,
            **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        net_out : list[int], optional
            Shape of the network's output, required only if layer contains factor
        batch_norm : bool, default = False
            If batch normalisation should be used
        activation : bool, default = True
            If ELU activation should be used
        kernel : int | list[int], default = 3
            Size of the kernel
        stride : int | list[int], default = 1
            Stride of the kernel
        out_padding : int, default = 0
            Padding applied to the output
        dropout : float, default =  0.1
            Probability of dropout

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        self._activation: bool = activation
        self._batch_norm: bool = batch_norm
        self._dropout: float = dropout
        self._slice: np.ndarray = np.array([slice(None)] * len(shapes[-1][1:]))
        shape: list[int] = shapes[-1].copy()
        transpose: nn.Module
        dropout_: nn.Module
        batch_norm_: nn.Module

        if isinstance(padding, str) and padding != 'same':
            raise ValueError(f'Transposed padding of {padding} in layer {idx} is unknown, padding '
                             f"must be either 'same', int, or list[int]")

        if ((np.array(out_padding) >= np.array(stride)) *
            (np.array(out_padding) >= np.array(dilation))).any():
            raise ValueError(f'Transposed output padding of {out_padding} in layer {idx} must be '
                             f'smaller than either the stride of {stride} or the dilation of '
                             f'{dilation}')

        if not 1 < len(shape) < 5:
            raise ValueError(f'Transposed in layer {idx} does not support tensors with more than 4 '
                             f'dimensions or less than 2, input shape is {shape}')

        if factor is not None and net_out is not None:
            shape[0] = max(1, int(net_out[0] * factor))
        elif filters:
            shape[0] = filters
        else:
            raise ValueError(f'Transposed in layer {idx} requires either factor or filters')

        transpose, dropout_, batch_norm_ = [
            [nn.ConvTranspose1d, nn.Dropout1d, nn.BatchNorm1d],
            [nn.ConvTranspose2d, nn.Dropout2d, nn.BatchNorm2d],
            [nn.ConvTranspose3d, nn.Dropout3d, nn.BatchNorm3d],
        ][len(shape) - 2]

        if padding == 'same':
            padding = _padding_transpose(kernel, stride, dilation, shapes[-1], shape)

        assert not isinstance(padding, str)
        shape = _kernel_transpose_shape(kernel, stride, padding, dilation, shapes[-1])

        if padding == 'same' and shape != shapes[-1]:
            self._slice[np.array(shape[1:]) - np.array(shapes[-1][1:]) == 1] = slice(-1, None)
            shape = shapes[-1].copy()

        self.layers.append(transpose(
            in_channels=shapes[-1][0],
            out_channels=shape[0],
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            output_padding=out_padding,
            dilation=dilation,
        ))

        # Optional layers
        if self._activation:
            self.layers.append(nn.ELU())

        if self._batch_norm:
            self.layers.append(batch_norm_(shape[0]))

        if self._dropout:
            self.layers.append(dropout_(self._dropout))

        shapes.append(shape)

    def forward(self, x: Tensor, **_: Any) -> Tensor:
        """
        Forward pass of the transposed convolutional layer

        Parameters
        ----------
        x : Tensor
            Input tensor with batch size N

        Returns
        -------
        Tensor
            Output tensor with batch size N
        """
        x = super().forward(x)
        return x[..., *self._slice]

class ConvTransposeUpscale(ConvTranspose):
    """
    Constructs an upscaler using a transposed convolutional layer.

    Supports 1D, 2D, and 3D transposed convolutional upscaling.

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            idx: int,
            shapes: list[list[int]],
            filters: int | None = None,
            factor: float | None = None,
            net_out: list[int] | None = None,
            batch_norm: bool = False,
            activation: bool = True,
            scale: int = 2,
            out_padding: int = 0,
            dropout: float = 0.1,
            **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        net_out : list[int], optional
            Shape of the network's output, required only if layer contains factor
        batch_norm : bool, default = False
            If batch normalisation should be used
        activation : bool, default = True
            If ELU activation should be used
        scale : int, default = 2
            Stride and size of the kernel, which acts as the upscaling factor
        out_padding : int, default = 0
            Padding applied to the output
        dropout : float, default =  0.1
            Probability of dropout

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            idx=idx,
            shapes=shapes,
            filters=filters,
            factor=factor,
            net_out=net_out,
            batch_norm=batch_norm,
            activation=activation,
            kernel=scale,
            stride=scale,
            padding=0,
            out_padding=out_padding,
            dropout=dropout,
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
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            idx: int,
            shapes: list[list[int]],
            filters: int | None = None,
            factor: float | None = None,
            net_out: list[int] | None = None,
            batch_norm: bool = False,
            activation: bool = True,
            scale: int = 2,
            kernel: int | list[int] = 3,
            dropout: float = 0,
            **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        net_out : list[int], optional
            Shape of the network's output, required only if layer contains factor
        batch_norm : bool, default = False
            If batch normalisation should be used
        activation : bool, default = True
            If ELU activation should be used
        scale : int, default = 2
            Factor to upscale the input by
        kernel : int | list[int], default = 3
            Size of the kernel
        dropout : float, default =  0
            Probability of dropout

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        filters_scale: int
        filters_scale = scale ** (len(shapes[-1]) - 1)

        if factor:
            filters = int(max(1, shapes[-1][0] * factor) * filters_scale)
        elif filters:
            filters = int(filters * filters_scale)
        else:
            raise ValueError(f'Convolution upscale in layer {idx} requires either factor or '
                             f'filters')

        # Convolutional layer
        super().__init__(
            idx=idx,
            shapes=shapes,
            filters=filters,
            net_out=net_out,
            batch_norm=batch_norm,
            activation=activation,
            stride=1,
            dropout=dropout,
            kernel=kernel,
            padding='same',
            **kwargs,
        )

        # Upscaling done using pixel shuffling
        self.layers.append(PixelShuffle(scale))
        shapes[-1][0] = shapes[-1][0] // filters_scale
        shapes[-1][1:] = [length * scale for length in shapes[-1][1:]]


class PixelShuffle(BaseLayer):
    r"""
    Used for upscaling by scale factor :math:`r` for an input :math:`(N,C\times r^n,D_1,...,D_n)` to
    an output :math:`(N,C,D_1\times r,...,D_n\times r)`

    Equivalent to :class:`torch.nn.PixelShuffle`, but for nD

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of pixel shuffle
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            scale: int,
            idx: int = 0,
            shapes: list[list[int]] | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        scale : int
            Upscaling factor
        idx : int, default = 0
            Layer number
        shapes : list[list[int]], default = None
            Shape of the outputs from each layer

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        self._scale: int = scale
        filters_scale: int

        # If not used as an individual layer in Network
        if not shapes:
            return

        filters_scale = self._scale ** (len(shapes[-1][1:]))

        if shapes[-1][0] % filters_scale != 0:
            raise ValueError(f'Pixel shuffle filters of {shapes[-1]} in layer {idx} must be an '
                             f'integer multiple of {filters_scale}')

        shapes.append(shapes[-1].copy())
        shapes[-1][0] = shapes[-1][0] // filters_scale
        shapes[-1][1:] = [length * self._scale for length in shapes[-1][1:]]

    def forward(self, x: Tensor, **_: Any) -> Tensor:
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
    strides, kernel, padding, dilation = _int_list_conversion(
        len(shape[1:]),
        [strides, kernel, padding, dilation]
    )

    for i, (stride, kernel_length, pad, dilation_length, length) in enumerate(zip(
        strides,
        kernel,
        padding,
        dilation,
        shape[1:]
    )):
        shape[i + 1] = max(
            1,
            stride * (length - 1) + dilation_length * (kernel_length - 1) - 2 * pad + 1
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
