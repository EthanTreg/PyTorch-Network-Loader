"""
Pooling network layers
"""
from typing import Any, Type, Literal, cast

from torch import nn, Tensor

from netloader.layers.misc import Pad
from netloader.layers.base import BaseSingleLayer
from netloader.utils import Shapes, compare_versions
from netloader.layers.utils import _kernel_shape, _padding


class AdaptivePool(BaseSingleLayer):
    """
    Uses pooling to downscale the input to the desired shape.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : nn.Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            shape: int | list[int],
            shapes: Shapes,
            *,
            channels: bool = True,
            mode: Literal['average', 'max'] = 'average',
            **kwargs: Any) -> None:
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

        self._check_pool_shape(shapes[-1])
        shape = self._check_adapt_pool(shapes[-1], shape)
        self._check_options('mode', mode, {None, 'average', 'max'})

        adapt_pool = [
            [nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d],
            [nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d],
            [nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d],
        ][len(shapes[-1]) - self._channels - 1][self._mode == 'average']
        self.layers.add_module('AdaptivePool', adapt_pool(shape))

        assert isinstance(shape, list)
        shapes.append(shapes[-1].copy())
        shapes[-1][self._channels:] = shape

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'channels': self._channels,
            'mode': self._mode,
            'shape': self.layers.AdaptivePool.output_size,  # type: ignore[union-attr]
        }

    def _check_pool_shape(self, shape: list[int]) -> None:
        """
        Checks if the input shape has more than 3 dimensions + channels or fewer than 1 + channels.

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
        shape.

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


class Pool(BaseSingleLayer):
    """
    Constructs a max or average pooling layer.

    Supports 1D, 2D, and 3D pooling.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : nn.Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the pool layer
    """
    def __init__(
            self,
            shapes: Shapes,
            *,
            kernel: int | list[int] = 2,
            stride: int | list[int] = 2,
            padding: int | Literal['same'] | list[int] = 0,
            mode: Literal['max', 'average'] = 'max',
            pool_kwargs: dict[str, Any] | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        shapes : list[list[int]]
            Shape of the outputs from each layer
        kernel : int | list[int], default = 2
            Size of the kernel
        stride : int | list[int], default = 2
            Stride of the kernel
        padding : int | {'same'} | list[int], default = 0
            Input padding, can an int or 'same' where 'same' preserves the input shape
        mode : {'max', 'average'}
            Whether to use 'max' or 'average' pooling
        pool_kwargs : dict[str, Any] | None, default = None
            Additional keyword arguments to pass to the pooling layer

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._mode: Literal['max', 'average'] = mode
        self._pad: int | Literal['same'] | list[int] = padding
        self._kwargs: dict[str, Any] = pool_kwargs or {}
        asymmetric: bool = False
        padding_: int | list[int]
        shape: list[int] = shapes[-1].copy()
        modes: tuple[str, str] = ('max', 'average')
        pool: Type[nn.Module]

        # Check for errors and calculate same padding
        if isinstance(self._pad, str):
            self._check_options('padding', self._pad, {'same'})
            self._check_stride(stride)
            asymmetric, padding_ = _padding(kernel, stride, shape, shape)
        else:
            padding_ = self._pad

        if asymmetric and compare_versions(self._ver, '3.10.1'):
            assert isinstance(padding_, list)
            self.layers.add_module('Pad', Pad(tuple(padding_)))
            padding_ = 0

        # Check for errors
        self._check_shape((1, 4), shape)
        self._check_options('mode', self._mode, set(modes))

        pool = cast(list[list[Type[nn.Module]]], [
            [nn.MaxPool1d, nn.AvgPool1d],
            [nn.MaxPool2d, nn.AvgPool2d],
            [nn.MaxPool3d, nn.AvgPool3d],
        ])[len(shape) - 2][self._mode == modes[1]]

        if self._mode == modes[1]:
            self._kwargs = {'count_include_pad': False} | self._kwargs

        self.layers.add_module('Pool', pool(
            kernel_size=kernel,
            stride=stride,
            padding=padding_,
            **self._kwargs,
        ))

        if self._pad == 'same' or asymmetric:
            shapes.append(shape)
        else:
            shapes.append(_kernel_shape(kernel, stride, padding_, shape))

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'kernel': self.layers.Pool.kernel_size,  # type: ignore[union-attr]
            'stride': self.layers.Pool.stride,
            'padding': self._pad,
            'mode': self._mode,
            'pool_kwargs': self._kwargs,
        }

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        Forward pass of the pool layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N,...) and type float, where N is the batch size
        *args
            Optional arguments to pass to the parent forward method
        **kwargs
            Optional keyword arguments to pass to the parent forward method

        Returns
        -------
        Tensor
            Output tensor of shape (N,...) and type float
        """
        return super().forward(x, *args, **kwargs)


class PoolDownscale(Pool):
    """
    Downscales the input using pooling.

    Supports 1D, 2D, and 3D pooling.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : nn.Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            scale: int,
            shapes: Shapes,
            *,
            mode: Literal['max', 'average'] = 'max',
            **kwargs: Any) -> None:
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

    def __getstate__(self) -> dict[str, Any]:
        state: dict[str, Any] = super().__getstate__()
        state.pop('padding', None)
        return state
