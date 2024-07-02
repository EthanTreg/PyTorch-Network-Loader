"""
Convolutional network layers
"""
from typing import Any

import torch
from torch import nn, Tensor

from netloader.layers.utils import BaseLayer


class AdaptivePool(BaseLayer):
    """
    Uses pooling to downscale the input to the desired shape

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the adaptive pool layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            idx: int,
            shape: int | list[int],
            shapes: list[list[int]],
            channels: bool = True,
            mode: str = 'average',
            **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shape : int | list[int]
            Output shape of the layer
        shapes : list[list[int]]
            Shape of the outputs from each layer
        channels : bool, default = True
            If the input includes a channels dimension
        mode : {'average', 'max'}
            Whether to use 'max' or 'average' pooling

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        self._channels: bool = channels
        self._out_shape: int | list[int] = shape
        self._mode: str = mode
        adapt_pool: nn.Module

        if len(shapes[-1]) - self._channels > 3 or len(shapes[-1]) - self._channels < 1:
            raise ValueError(f'Adaptive pool in layer {idx} does not support tensors with more '
                             f'than 3 dimensions + channels or less than 1 + channels, input shape '
                             f'is {shapes[-1]} and channels is {bool(self._channels)}.')

        adapt_pool = [
            [nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d],
            [nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d],
            [nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d],
        ][len(shapes[-1]) - self._channels - 1][self._mode == 'average']
        self.layers.append(adapt_pool(self._out_shape))

        if isinstance(self._out_shape, int):
            self._out_shape = [self._out_shape] * (len(shapes[-1]) - self._channels)
        elif len(self._out_shape) == 1:
            self._out_shape = self._out_shape * (len(shapes[-1]) - self._channels)
        elif len(self._out_shape) != len(shapes[-1]) - self._channels:
            raise ValueError(
                f'Target output shape {self._out_shape} in layer {idx} does not match the input '
                f'shape {shapes[-1]} if channels is {bool(self._channels)}, output shape must be '
                f'either 1, or {len(shapes[-1]) - self._channels}'
            )

        shapes.append(shapes[-1].copy())
        shapes[-1][self._channels:] = self._out_shape

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'channels={bool(self._channels)}, mode={self._mode}'


class Conv(BaseLayer):
    """
    Convolutional layer constructor

    Supports 1D and 2D convolution

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
            stride: int | list[int] = 1,
            kernel: int | list[int] = 3,
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
        stride : int | list[int], default = 1
            Stride of the kernel
        kernel : int | list[int], default = 3
            Size of the kernel
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
            raise ValueError(f'Unknown padding: {padding} in layer {idx}, padding must be either '
                             f"'same' or an int")

        if padding != 'same' and (
                torch.tensor(shape[1:]) + torch.tensor(padding) < torch.tensor(kernel)
        ).any():
            raise ValueError(f'Kernel size {kernel} in layer {idx} is too large for input shape '
                             f'{shape} and padding {padding}')

        if (torch.tensor(stride) > 1).any() and padding == 'same':
            raise ValueError(f'Same padding in layer {idx} is not supported for strided '
                             f'convolution')

        if len(shape) > 4 or len(shape) < 2:
            raise ValueError(f'Convolution in layer {idx} does not support tensors with more than '
                             f'4 dimensions or less than 2, input shape is {shape}')

        if factor is not None and net_out is not None:
            shape[0] = max(1, int(net_out[0] * factor))
        elif filters:
            shape[0] = filters
        else:
            raise ValueError(f'Either factor or filters is required as input in layer {idx}')

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
        elif isinstance(padding, str):
            raise ValueError(f"Unknown padding type in layer {idx}, padding must be either 'same' "
                             f"| int | list[int].")
        else:
            shapes.append(_kernel_shape(stride, kernel, padding, shape))


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

    Supports 1D and 2D transposed convolution

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
        super().__init__(idx=idx, **kwargs)
        self._activation: bool = activation
        self._batch_norm: bool = batch_norm
        self._dropout: float = dropout
        shape: list[int] = shapes[-1].copy()
        transpose: nn.Module
        dropout_: nn.Module
        batch_norm_: nn.Module

        if out_padding > scale:
            raise ValueError(f'Output padding of {out_padding} in layer {idx} must be smaller than '
                             f'the scale factor {scale}')

        if len(shape) > 4 or len(shape) < 2:
            raise ValueError(f'Transposed convolution in layer {idx} does not support tensors with '
                             f'more than 4 dimensions or less than 2, input shape is {shape}')

        if factor is not None and net_out is not None:
            shape[0] = max(1, int(net_out[0] * factor))
        elif filters:
            shape[0] = filters
        else:
            raise ValueError(f'Either factor or filters is required as input in layer {idx}')

        transpose, dropout_, batch_norm_ = [
            [nn.ConvTranspose1d, nn.Dropout1d, nn.BatchNorm1d],
            [nn.ConvTranspose2d, nn.Dropout2d, nn.BatchNorm2d],
            [nn.ConvTranspose3d, nn.Dropout3d, nn.BatchNorm3d],
        ][len(shape) - 2]

        self.layers.append(transpose(
            in_channels=shapes[-1][0],
            out_channels=shape[0],
            kernel_size=scale,
            stride=scale,
            output_padding=out_padding,
        ))

        # Optional layers
        if self._activation:
            self.layers.append(nn.ELU())

        if self._batch_norm:
            self.layers.append(batch_norm_(shape[0]))

        if self._dropout:
            self.layers.append(dropout_(self._dropout))

        # Data size doubles
        shape[1:] = [length * scale + out_padding for length in shape[1:]]
        shapes.append(shape)


class ConvUpscale(Conv):
    """
    Constructs an upscaler using a convolutional layer and pixel shuffling.

    Supports 1D and 2D convolutional upscaling.

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
            raise ValueError(f'Either factor or filters is required as input in layer {idx}')

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

        if not shapes:
            return

        filters_scale = self._scale ** (len(shapes[-1]) - 1)

        if shapes[-1][0] % filters_scale != 0:
            raise ValueError(f'Number of filters {shapes[-1]} in layer {idx} must be an int '
                             f'multiple of {filters_scale}')

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


class Pool(BaseLayer):
    """
    Constructs a max or average pooling layer

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            idx: int,
            shapes: list[list[int]],
            kernel: int | list[int] = 2,
            stride: int | list[int] = 2,
            padding: int | str | list[int] = 0,
            mode: str = 'max',
            **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        kernel : int | list[int], default = 2
            Size of the kernel
        stride : int | list[int], default = 2
            Stride of the kernel
        padding : int | str | list[int], default = 0
            Input padding, can an int or 'same' where 'same' preserves the input shape
        mode : {'max', 'average'}
            Whether to use 'max' or 'average' pooling

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        self._mode: str
        avg_kwargs: dict[str, bool]
        pool: nn.Module

        if len(shapes[-1]) > 4 or len(shapes[-1]) < 2:
            raise ValueError(f'Pooling in layer {idx} does not support tensors with more than 4 '
                             f'dimensions or less than 2, input shape is {shapes[-1]}')

        self._mode = mode
        avg_kwargs = {}
        pool = [
            [nn.MaxPool1d, nn.AvgPool1d],
            [nn.MaxPool2d, nn.AvgPool2d],
            [nn.MaxPool3d, nn.AvgPool3d],
        ][len(shapes[-1]) - 2][self._mode == 'average']

        if padding == 'same':
            padding = _padding(kernel, stride, shapes[-1], shapes[-1])

        if self._mode == 'average':
            avg_kwargs = {'count_include_pad': False}

        assert not isinstance(padding, str)
        self.layers.append(pool(kernel_size=kernel, stride=stride, padding=padding, **avg_kwargs))
        shapes.append(_kernel_shape(stride, kernel, padding, shapes[-1].copy()))


class PoolDownscale(Pool):
    """
    Downscales the input using pooling
    
    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass
    """
    def __init__(self, scale: int, shapes: list[list[int]], mode: str = 'max', **kwargs: Any):
        """        
        Parameters
        ----------
        scale : int
            Stride and size of the kernel, which acts as the downscaling factor
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        mode : {'max', 'average'}
            Whether to use 'max' or 'average' pooling

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(shapes=shapes, kernel=scale, stride=scale, padding=0, mode=mode, **kwargs)


def _kernel_shape(
        strides: int | list[int],
        kernel: int | list[int],
        padding: int | list[int],
        shape: list[int]) -> list[int]:
    """
    Calculates the output shape after a kernel operation

    Parameters
    ----------
    strides : int | list[int]
        Stride of the kernel
    kernel : int | list[int]
        Size of the kernel
    padding : int | list[int]
        Input padding
    shape : list[int]
        Input shape into the layer

    Returns
    -------
    list[int]
        Output shape of the layer
    """
    if isinstance(strides, int):
        strides = [strides] * len(shape[1:])

    if isinstance(kernel, int):
        kernel = [kernel] * len(shape[1:])

    if isinstance(padding, int):
        padding = [padding] * len(shape[1:])

    for i, (stride, kernel_length, pad, length) in enumerate(zip(
            strides,
            kernel,
            padding,
            shape[1:])
    ):
        shape[i + 1] = max(1, (length + 2 * pad - kernel_length) // stride + 1)

    return shape


def _padding(
        kernel: int | list[int],
        strides: int | list[int],
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
    in_shape : list[int]
        Input shape into the layer
    out_shape : list[int]
        Output shape from the layer

    Returns
    -------
    list[int]
        Required padding for specific output shape
    """
    padding: list[int] = []

    if isinstance(strides, int):
        strides = [strides] * len(in_shape[1:])

    if isinstance(kernel, int):
        kernel = [kernel] * len(in_shape[1:])

    for stride, kernel_length, in_length, out_length in zip(
            strides,
            kernel,
            in_shape[1:],
            out_shape[1:],
    ):
        padding.append((stride * (out_length - 1) + kernel_length - in_length) // 2)

    return padding
