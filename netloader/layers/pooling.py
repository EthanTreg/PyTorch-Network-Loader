"""
Pooling network layers
"""
from typing import Any

import numpy as np
from torch import nn

from netloader.layers.utils import BaseLayer, _kernel_shape, _padding


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

        if isinstance(self._out_shape, int):
            self._out_shape = [self._out_shape] * (len(shapes[-1]) - self._channels)
        elif len(self._out_shape) == 1:
            self._out_shape = self._out_shape * (len(shapes[-1]) - self._channels)
        elif len(self._out_shape) != len(shapes[-1]) - self._channels:
            raise ValueError(f'Adaptive pool target shape {self._out_shape} in layer {idx} does '
                             f'not match the input shape {shapes[-1]} if channels is '
                             f'{bool(self._channels)}, output dimensions must be either 1, or '
                             f'{len(shapes[-1]) - self._channels}')

        adapt_pool = [
            [nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d],
            [nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d],
            [nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d],
        ][len(shapes[-1]) - self._channels - 1][self._mode == 'average']
        self.layers.append(adapt_pool(self._out_shape))

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


class Pool(BaseLayer):
    """
    Constructs a max or average pooling layer.

    Supports 1D, 2D, and 3D pooling.

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

        if isinstance(padding, str) and padding != 'same':
            raise ValueError(f'Pooling padding of {padding} in layer {idx} is unknown, padding '
                             f"must be either 'same', int, or list[int]")

        if (np.array(stride) > 1).any() and padding == 'same':
            raise ValueError(f"Pooling 'same' padding in layer {idx} is not supported for "
                             f'strides > 1, stride is {stride}')

        if not 1 < len(shapes[-1]) < 5:
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
        shapes.append(_kernel_shape(kernel, stride, padding, shapes[-1].copy()))


class PoolDownscale(Pool):
    """
    Downscales the input using pooling.

    Supports 1D, 2D, and 3D pooling.

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
