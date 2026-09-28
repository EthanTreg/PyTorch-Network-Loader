"""
Miscellaneous network layers
"""
from copy import deepcopy
from warnings import warn
from typing import Any, Self, cast, overload

import torch
import numpy as np
from torch import nn, Tensor

from netloader.utils import Shapes
from netloader.data import DataList
from netloader.layers import multi_layer
from netloader.utils.types import TensorListLike
from netloader.layers.base import BaseLayer, BaseSingleLayer


class BatchNorm(BaseSingleLayer):
    """
    Batch Normalisation layer

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    layers : Sequential
        Layers to loop through in the forward pass
    """
    def __init__(self, shapes: Shapes, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        shapes : Shapes
            Shape of the outputs from each layer

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._check_shape((1, 4), shapes[-1])
        self.layers.add_module(
            'BatchNorm',
            [
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ][max(0, len(shapes[-1]) - 2)](shapes[-1][0]),
        )
        shapes.append(shapes[-1].copy())


class Checkpoint(BaseLayer):
    """
    Constructs a layer to create a checkpoint for using the output from the previous layer in
    future layers.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    """
    def __init__(
            self,
            shapes: Shapes,
            check_shapes: Shapes,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        shapes : Shapes
            Shape of the outputs from each layer
        check_shapes : Shapes
            Shape of the outputs from each checkpoint
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._check_num: int = len(check_shapes)
        shapes.append(deepcopy(shapes.get(-1, True)))
        check_shapes.append(deepcopy(shapes.get(-1, True)))

    def forward(
            self,
            x: TensorListLike,
            checkpoints: list[TensorListLike],
            *_: Any,
            **__: Any) -> TensorListLike:
        """
        Forward pass of the checkpoint layer.

        Parameters
        ----------
        x : TensorListLike
            Input with tensors of shape (N,...) and type float, where N is the batch size
        checkpoints : list[TensorListLike]
            List of checkpoint values with tensors of shape (N,...) and type float

        Returns
        -------
        TensorListLike
            Output with tensors of shape (N,...) and type float
        """
        checkpoints.append(x.clone())
        return x

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'checkpoint_num={self._check_num}'


class Dropout(BaseSingleLayer):
    """
    Constructs a dropout layer to randomly zero out elements of the input tensor.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    layers : Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            prob: float,
            *,
            shapes: Shapes | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        prob : float
            Probability of dropout
        shapes : Shapes | None, Optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self.layers.add_module('Dropout', nn.Dropout(prob))

        # If not used as a layer in a network
        if not shapes:
            return

        shapes.append(shapes[-1].copy())

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'prob': self.layers.Dropout.p}  # type: ignore[union-attr]


class DropPath(BaseLayer):
    """
    Constructs a drop path layer to drop samples in a batch.

    See `FractalNet: Ultra-Deep Neural Networks without Residuals
    <https://arxiv.org/abs/1605.07648>`_ by Larsson et al. (2016) for details.

    Custom version of implementation by `timm
    <https://github.com/pprp/timm/blob/master/timm/layers/drop.py#L157>`_.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    """
    def __init__(
            self,
            prob: float,
            *,
            shapes: Shapes | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        prob : float
            Probability of dropout
        shapes : Shapes | None, Optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._keep_prob: float = 1 - prob

        # If not used as a layer in a network
        if not shapes:
            return

        shapes.append(shapes[-1].copy())

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'prob': 1 - self._keep_prob}

    def forward(self, x: Tensor, *_: Any, **__: Any) -> Tensor:
        """
        Forward pass of the drop path layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N,...) and type float, where N is the batch size

        Returns
        -------
        Tensor
            Output tensor of shape (N,...) and type float
        """
        if not self.training or self._keep_prob == 1:
            return x

        mask: Tensor = x.new_empty(x.shape[0:1] + (1,) * (x.ndim - 1)).bernoulli_(self._keep_prob)

        if self._keep_prob:
            mask.div_(self._keep_prob)

        return x * mask

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'prob={1 - self._keep_prob}'


class Index(BaseLayer):
    """
    Constructs a layer to slice the last dimension from the output from the previous layer.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    """
    @overload
    def __init__(
            self,
            *,
            index: int | list[int | None] | tuple[int | None, ...] | slice,
            map_: bool = ...,
            shapes: Shapes | None = ...,
            **kwargs: Any) -> None: ...

    @overload
    def __init__(
            self,
            number: int,
            *,
            map_: bool = ...,
            greater: bool = ...,
            shapes: Shapes | None = ...,
            **kwargs: Any) -> None: ...

    def __init__(
            self,
            *args: int,
            map_: bool = True,
            index: int | list[int | None] | tuple[int | None, ...] | slice | None = None,
            shapes: Shapes | None = None,
            **kwargs: Any) -> None:
        r"""
        Parameters
        ----------
        index : int | list[int | None] | tuple[int | None, ...] | slice
            Index or slice to apply to the last dimension of the input tensor(s) or if map\_ is
            False, the DataList itself
        map\_ : bool, Optional
            If slicing should be applied to each tensor in a DataList, or if the DataList should be
            sliced itself, only used if input is a DataList, default = True
        shapes : Shapes | None, Optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        number: int
        shape: list[int]
        in_shape: list[int] | list[list[int]]
        self._map: bool = map_
        self._idx: int | slice

        if args or 'number' in kwargs:
            warn(
                'number argument and greater keyword argument are deprecated, please use '
                'index instead',
                DeprecationWarning,
                stacklevel=2,
            )
            number = kwargs.pop('number') if 'number' in kwargs else args[0]
            self._idx = slice(number, None) if kwargs.pop('greater', True) else slice(number)
        elif index is None and 'idx' in kwargs:
            warn(
                'idx argument is deprecated, please use index instead',
                DeprecationWarning,
                stacklevel=2,
            )
            self._idx = kwargs['idx']
        else:
            assert index is not None
            self._idx = slice(*index) if isinstance(index, (tuple, list)) else index

        # If not used as a layer in a network
        if not shapes:
            return

        in_shape = deepcopy(shapes.get(-1, True))

        # Length of slice
        if isinstance(in_shape[0], int):
            in_shape = self._shape_index(cast(list[int], in_shape))
        elif isinstance(in_shape[0], list) and self._map:
            in_shape = [self._shape_index(shape) for shape in cast(list[list[int]], in_shape)]
        else:
            in_shape = cast(list[list[int]], in_shape)[self._idx]

        shapes.append(in_shape)

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'map_': self._map, 'index': self._idx}

    def _shape_index(self, shape: list[int]) -> list[int]:
        """
        Returns the output shape after indexing.

        Parameters
        ----------
        shape : list[int]
            Input shape

        Returns
        -------
        list[int]
            Output shape
        """
        out_shape: list[int] = shape.copy()
        idx: tuple[int, int, int]

        if isinstance(self._idx, int):
            out_shape[-1] = 1
        else:
            idx = self._idx.indices(shape[-1])
            out_shape[-1] = (idx[1] - idx[0] + (idx[2] - 1)) // idx[2]
        return out_shape

    def forward(self, x: TensorListLike, *_: Any, **__: Any) -> TensorListLike:
        """
        Forward pass of the indexing layer.

        Parameters
        ----------
        x : TensorListLike
            Input tensor(s) of shape (N,...) and type float, where N is the batch size

        Returns
        -------
        TensorListLike
            Output tensor(s) of shape (N,...) and type float
        """
        if isinstance(x, DataList) and not self._map:
            return x.get(self._idx, list_=True) if isinstance(self._idx, int) else (
                x.get(self._idx, list_=True))
        return x[..., self._idx] if isinstance(x, Tensor) else x[self._idx]

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'map={bool(self._map)}, idx={self._idx}'


class LayerNorm(BaseLayer):
    """
    Constructs a layer normalisation layer that is the same as PyTorch's LayerNorm, except
    normalisation priority is the first dimension after batch dimension to follow ConvNeXt
    implementation.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    """
    def __init__(
            self,
            *,
            dims: int | None = None,
            shape: list[int] | None = None,
            shapes: Shapes | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        dims : int | None, Optional
            Number of dimensions to normalise starting with the first dimension, ignoring batch
            dimension, requires shapes argument, won't be used if shape is provided
        shape : list[int] | None, Optional
            Input shape or shape of the first dimension to normalise, will be used if provided, else
            dims will be used
        shapes : Shapes | None, Optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._layer: nn.LayerNorm

        if not shape and (dims and shapes):
            shape = shapes[-1][:dims]
        elif not shape:
            raise ValueError(f'Either shape ({shape}) or dims ({dims}) and shapes must be provided')

        self._layer = nn.LayerNorm(shape[::-1])

        # If not used as a layer in a network
        if not shapes:
            return

        shapes.append(shapes[-1].copy())

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'shape': self._layer.normalized_shape}

    def forward(self, x: Tensor, *_: Any, **__: Any) -> Tensor:
        """
        Forward pass of the layer normalisation layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N,...) and type float, where N is the batch size

        Returns
        -------
        Tensor
            Output Tensor of shape (N,...) and type float
        """
        permutation: list[int] = list(range(x.ndim - 1, 0, -1))
        return self._layer(x.permute(0, *permutation)).permute(0, *permutation)


class Pad(BaseLayer):
    """
    Constructs a padding layer to pad the output from the previous layer.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    """
    @overload
    def __init__(
            self,
            pad: list[int] | list[tuple[int, int]] | tuple[int, ...],
            *,
            shapes: Shapes | None = ...,
            pad_kwargs: dict[str, Any] | None = ...,
            **kwargs: Any) -> None: ...

    @overload
    def __init__(
            self,
            pad: int,
            *,
            dims: int = ...,
            shapes: Shapes = ...,
            pad_kwargs: dict[str, Any] | None = ...,
            **kwargs: Any) -> None: ...

    def __init__(
            self,
            pad: int | list[int] | list[tuple[int, int]] | tuple[int, ...],
            *,
            dims: int = -1,
            shapes: Shapes | None = None,
            pad_kwargs: dict[str, Any] | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        pad : int | list[int] | list[tuple[int, int]] | tuple[int, ...]
            Amount of padding to apply to each dimension of the input tensor, ignoring batch
            dimension and from right to left, if int is provided, same padding will be applied to
            left and right of all dimensions according to dims, if list of integers is provided,
            each integer will be applied to left and right of the corresponding dimension, if list
            of tuple of integers is provided, then asymmetric (left, right) padding will be applied,
            if tuple of integers is provided, each pair of integers will be applied to left and
            right of the corresponding dimension
        dims : int, Optional
            Number of dimensions to pad starting with the last dimension, ignoring batch dimension,
            if shapes is not provided, dims must be greater than 0, if negative, excludes the first
            abs(dims) dimensions, if 0 pads all dimensions, won't be used if padding is a list of
            integers or list of tuple of integers, default = -1
        shapes : Shapes | None, Optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary
        pad_kwargs : dict[str, Any] | None, Optional
            Additional keyword arguments to pass to the padding function
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._pad: tuple[int, ...]
        self._kwargs: dict[str, Any] = pad_kwargs or {}
        pad_: list[int]
        sub_shape: list[int]
        shape: list[int] | list[list[int]]

        if isinstance(pad, int) and dims <= 0 and shapes is None:
            raise ValueError(f'Dims ({dims}) must be greater than 0 if pad ({pad}) is an integer '
                             f'and shapes is not provided')

        if shapes is None and dims <= 0:
            dims = 1

        if isinstance(pad, tuple) and len(pad) % 2 != 0:
            raise ValueError(f'If pad is a tuple, it must contain an even number of integers '
                             f'({len(pad)})')

        if dims < 0 and shapes:
            dims = len(shapes[-1]) + dims
        elif dims == 0 and shapes:
            len(shapes[-1])

        if isinstance(pad, int):
            self._pad = tuple([pad] * 2 * dims)
        elif isinstance(pad, list) and isinstance(pad[0], int):
            self._pad = tuple(np.repeat(pad, 2).tolist())
        elif isinstance(pad, list):
            self._pad = tuple(np.array(pad).flatten().tolist())
        else:
            self._pad = pad

        if shapes is None:
            return

        shape = shapes.get(-1, True)
        pad_ = ([0] * (len(shape) - len(self._pad) // 2) +
                [sum(self._pad[i - 2:i]) for i in range(len(self._pad), 0, -2)])

        if isinstance(shape[0], list):
            shapes.append([
                self._pad_shape(cast(list[int], sub_shape), pad_) for sub_shape in shape
            ])
        else:
            shapes.append(self._pad_shape(cast(list[int], shape), pad_))

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'pad': self._pad,
            'pad_kwargs': self._kwargs,
        }

    @staticmethod
    def _pad_shape(shape: list[int], dim_pads: list[int]) -> list[int]:
        """
        Returns the output shape after padding.

        Parameters
        ----------
        shape : list[int]
            Input shape
        dim_pads : list[int]
            Amount of padding applied to each dimension

        Returns
        -------
        list[int]
            Output shape
        """
        dim: int
        dim_pad: int
        return [dim + dim_pad for dim, dim_pad in zip(shape, dim_pads)]

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'pad={list(zip(self._pad[::2], self._pad[1::2]))}, pad_kwargs={self._kwargs}'

    def forward(self, x: TensorListLike, *_: Any, **__: Any) -> TensorListLike:
        """
        Forward pass of the padding layer.

        Parameters
        ----------
        x : TensorListLike
            Input with tensor(s) of shape (N,...) and type float, where N is the batch size

        Returns
        -------
        TensorListLike
            Output with tensor(s) of shape (N,...) and type float
        """
        if isinstance(x, Tensor):
            return nn.functional.pad(x, self._pad, **self._kwargs)
        return x.apply(nn.functional.pad, self._pad, **self._kwargs)


class Reshape(BaseLayer):
    """
    Constructs a reshaping layer to change the data dimensions.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    """
    def __init__(
            self,
            shape: list[int],
            *,
            factor: bool = False,
            layer: int | None = None,
            net_out: list[int] | None = None,
            shapes: Shapes | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        shape : list[int]
            Desired shape of the output tensor, ignoring first dimension
        factor : bool, Optional
            If reshape should be relative to the network output shape, or if layer is provided,
            which layer to be relative to, requires tracking layer outputs, default = False
        layer : int | None, Optional
            If factor is True, which layer for factor to be relative to, if None, network output
            will be used
        net_out : list[int] | None, Optional
            Shape of the network's output, required only if factor is True
        shapes : Shapes | None, Optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._shape: list[int] = shape
        prod: int
        elm: int
        target: list[int]

        # If not used as a layer in a network
        if shapes is None:
            return

        if factor and shapes is not None and net_out is not None:
            target = shapes[layer] if layer is not None else net_out
            self._shape = [
                int(elm * length) if elm != -1 else -1 for elm, length in zip(self._shape, target)
            ]
        if factor and net_out is None:
            raise ValueError('factor requires net_out to not be None')

        # If -1 in output shape, calculate the dimension length from the input dimensions
        if -1 not in self._shape:
            shapes.append(self._shape)
        elif self._shape.count(-1) == 1:
            shape = self._shape.copy()
            prod = np.prod(np.array(shape)[np.array(shape) != -1])
            shape[shape.index(-1)] = int(np.prod(shapes[-1]) // prod)
            shapes.append(shape)
        else:
            raise ValueError(f'Cannot infer output shape as -1 occurs more than once in '
                             f'{self._shape}')

        # If input tensor cannot be reshaped into output shape
        self._check_reshape(shapes[-2], shapes[-1])

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'shape': self._shape}

    @staticmethod
    def _check_reshape(in_shape: list[int], out_shape: list[int]) -> None:
        """
        Checks if the input tensor can be reshaped into the output tensor.

        Parameters
        ----------
        in_shape : list[int]
            Input shape
        out_shape : list[int]
            Output shape
        """
        if np.prod(out_shape) != np.prod(in_shape):
            raise ValueError(f'Input size does not match output size for input shape {in_shape} '
                             f'& output shape {out_shape}')

    def forward(self, x: Tensor, *_: Any, **__: Any) -> Tensor:
        """
        Forward pass of reshaping tensors.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N,...) and type float, where N is the batch size

        Returns
        -------
        Tensor
            Output tensor of shape (N,...) and type float
        """
        return x.contiguous().view(x.size(0), *self._shape)

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'output={self._shape}'


class Scale(BaseLayer):
    """
    Constructs a scaling layer that scales the output by a learnable tensor.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    """
    def __init__(
            self,
            dims: int,
            scale: float,
            shapes: Shapes,
            *,
            first: bool = True,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        dims : int
            Number of dimensions to have individual scales for
        scale : float
            Initial scale factor
        shapes : Shapes
            Shape of the outputs from each layer
        first : bool, Optional
            If dims should count from the first dimension after the batch dimension, or from the
            final dimension backwards, default = True
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._first: bool = first
        self._scale: Tensor
        shape: list[int]

        if self._first:
            shape = shapes[-1][:dims]
        else:
            shape = shapes[-1][-dims:]

        self._scale = nn.Parameter(scale * torch.ones(*shape), requires_grad=True).to(self._device)

        shapes.append(shapes[-1].copy())

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'first': self._first,
            'dims': self._scale.ndim,
            'scale': self._scale.detach().flatten().cpu().tolist()[0],
        }

    def extra_repr(self) -> str:
        return (f'first={self._first}, dims={self._scale.ndim}, '
                f'scale={self._scale.detach().flatten().cpu().tolist()[0]}')

    def forward(self, x: Tensor, *_: Any, **__: Any) -> Tensor:
        """
        Forward pass of the scale layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N,...) and type float, where N is the batch size

        Returns
        -------
        Tensor
            Output tensor of shape (N,...) and type float
        """
        if self._first:
            permutation: list[int] = list(range(x.ndim - 1, 0, -1))
            return (self._scale * x.permute(0, *permutation)).permute(0, *permutation)
        return self._scale * x

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self._scale = self._scale.to(*args, **kwargs)
        return self


__all__ = [
    'BatchNorm',
    'Checkpoint',
    'Dropout',
    'DropPath',
    'Index',
    'LayerNorm',
    'Pad',
    'Reshape',
    'Scale',
]


def __getattr__(name: str) -> Any:
    if hasattr(multi_layer, name):
        warn(
            f'Accessing {name} from layers.misc is deprecated, please use layers or '
            f'layers.multi_layer instead',
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(multi_layer, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
