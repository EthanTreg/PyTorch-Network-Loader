"""
Pooling network layers
"""
from typing import Any, Type

import numpy as np
from torch import nn, Tensor

from netloader.layers.utils import _kernel_shape, _padding
from netloader.layers.base import BaseLayer


class AdaptivePool(BaseLayer):
    """
    Uses pooling to downscale the input to the desired shape

    Attributes
    ----------
    layers : Sequential
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
            shape: int | list[int],
            shapes: list[list[int]],
            channels: bool = True,
            mode: str = 'average',
            **kwargs: Any):
        """
        Parameters
        ----------
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
        super().__init__(**kwargs)
        self._channels: bool = channels
        self._mode: str = mode
        adapt_pool: Type[nn.Module]

        self._check_shape(shapes[-1])
        self._check_adapt_pool(shapes[-1], shape)

        adapt_pool = [
            [nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d],
            [nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d],
            [nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d],
        ][len(shapes[-1]) - self._channels - 1][self._mode == 'average']
        self.layers.add_module('AdaptivePool', adapt_pool(shape))

        assert isinstance(shape, list)
        shapes.append(shapes[-1].copy())
        shapes[-1][self._channels:] = shape

    def _check_shape(self, shape: list[int]) -> None:
        """
        Checks if the input shape has more than 3 dimensions + channels or fewer than 1 + channels

        Parameters
        ----------
        shape : list[int]
            Input shape
        """
        if len(shape) - self._channels > 3 or len(shape) - self._channels < 1:
            raise ValueError(f'Tensors with more than 3 dimensions + channels or less than 1 + '
                             f'channels is not supported, input shape is {shape} and channels is '
                             f'{bool(self._channels)}.')

    def _check_adapt_pool(self, in_shape: list[int], shape: int | list[int]) -> list[int]:
        """
        Checks if the target shape is compatible with the input shape and calculates the output
        shape

        Parameters
        ----------
        in_shape : list[int]
            Input shape
        shape : int | list[int]
            Target shape

        Returns
        -------
        list[int]
            Output shape
        """
        if isinstance(shape, int):
            shape = [shape] * (len(in_shape) - self._channels)
        elif len(shape) == 1:
            shape = shape * (len(in_shape) - self._channels)
        elif len(shape) != len(in_shape) - self._channels:
            raise ValueError(f'Target shape {shape} does not match the input shape {in_shape} if '
                             f'channels is {bool(self._channels)}, output dimensions must be '
                             f'either 1, or {len(in_shape) - self._channels}')

        return shape

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'channels={bool(self._channels)}, mode={self._mode}'


class Pool(BaseLayer):
    """
    Constructs a max or average pooling layer.

    Supports 1D, 2D, and 3D pooling.

    Attributes
    ----------
    layers : Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            shapes: list[list[int]],
            kernel: int | list[int] = 2,
            stride: int | list[int] = 2,
            padding: int | str | list[int] = 0,
            mode: str = 'max',
            **kwargs: Any):
        """
        Parameters
        ----------
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
        super().__init__(**kwargs)
        self._pad: np.ndarray = np.zeros(2 * len(shapes[-1][1:]), dtype=int)
        padding_: int | str | list[int] = padding
        shape: list[int]
        modes: tuple[str, str] = ('max', 'average')
        avg_kwargs: dict[str, bool] = {}
        pool: Type[nn.Module]

        # Check for errors and calculate same padding
        if isinstance(padding, str):
            self._check_options('padding', padding, {'same'})
            self._check_stride(stride)
            padding = _padding(kernel, stride, shapes[-1], shapes[-1])

        # Check for errors
        self._check_shape(shapes[-1])
        self._check_options('mode', mode, set(modes))

        pool = [
            [nn.MaxPool1d, nn.AvgPool1d],
            [nn.MaxPool2d, nn.AvgPool2d],
            [nn.MaxPool3d, nn.AvgPool3d],
        ][len(shapes[-1]) - 2][mode == modes[1]]

        assert not isinstance(padding, str)
        shape = _kernel_shape(
            kernel,
            stride,
            padding,
            shapes[-1],
        )

        if mode == modes[1]:
            avg_kwargs = {'count_include_pad': False}

        # Correct same padding for one-sided padding
        if padding_ == 'same' and shape != shapes[-1]:
            self._pad[np.argwhere(
                np.array(shape[1:]) != np.array(shapes[-1][1:])
            ).flatten() * 2 + 1] = 1
            shape = shapes[-1].copy()

        assert not isinstance(padding, str)
        self.layers.add_module('Pool', pool(
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            **avg_kwargs,
        ))
        shapes.append(shape)

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """
        Forward pass of the pool layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor with batch size N

        **kwargs
            Arguments to be passed to parent forward method

        Returns
        -------
        (N,...) Tensor
            Output tensor with batch size N
        """
        x = nn.functional.pad(x, tuple(self._pad))
        return super().forward(x, **kwargs)


class PoolDownscale(Pool):
    """
    Downscales the input using pooling.

    Supports 1D, 2D, and 3D pooling.

    Attributes
    ----------
    layers : Sequential
        Layers to loop through in the forward pass
    """
    def __init__(self, scale: int, shapes: list[list[int]], mode: str = 'max', **kwargs: Any):
        """
        Parameters
        ----------
        scale : int
            Stride and size of the kernel, which acts as the downscaling factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        mode : {'max', 'average'}
            Whether to use 'max' or 'average' pooling

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(shapes=shapes, kernel=scale, stride=scale, padding=0, mode=mode, **kwargs)
