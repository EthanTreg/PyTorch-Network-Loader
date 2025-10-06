"""
Miscellaneous network layers
"""
from copy import deepcopy
from warnings import warn
from typing import Any, Self, cast

import torch
import numpy as np
from numpy import ndarray
from torch import nn, Tensor

from netloader.utils import Shapes
from netloader.data import DataList
from netloader.utils.types import TensorListLike
from netloader.layers.base import BaseLayer, BaseSingleLayer, BaseMultiLayer


class BatchNorm(BaseSingleLayer):
    """
    Batch Normalisation layer

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : nn.Sequential
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
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x, checkpoints) -> TensorListLike
        Forward pass of the checkpoint layer
    extra_repr() -> str:
        Displays layer parameters when printing the network
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
        shapes.append(shapes[-1].copy())
        check_shapes.append(shapes[-1].copy())

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


class Concatenate(BaseMultiLayer):
    """
    Constructs a concatenation layer to combine the outputs from two layers.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x, outputs, checkpoints) -> Tensor
        Forward pass of the concatenation layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            net_check: bool,
            layer: int,
            shapes: Shapes,
            check_shapes: Shapes,
            *,
            checkpoint: bool = False,
            dim: int = 0,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        layer : int
            Layer index to concatenate the previous layer output with
        shapes : Shapes
            Shape of the outputs from each layer
        check_shapes : Shapes
            Shape of the outputs from each checkpoint
        checkpoint : bool, default = False
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints
        dim : int, default = 0
            Dimension to concatenate to
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(net_check, layer, shapes, check_shapes, checkpoint=checkpoint, **kwargs)
        self._dim: int = dim
        shape: list[int] = shapes[-1].copy()

        # If tensors cannot be concatenated along the specified dimension
        self._check_concatenation(shape)

        shape[self._dim] = shape[self._dim] + self._target[self._dim]
        shapes.append(shape)

    def _check_concatenation(self, shape: list[int]) -> None:
        """
        Checks if input shape and target shape are compatible for concatenation.

        Parameters
        ----------
        shape : list[int]
            Input shape
        """
        if ((self._target[:self._dim] + (self._target[self._dim + 1:] if self._dim != -1 else []) !=
             shape[:self._dim] + (shape[self._dim + 1:] if self._dim != -1 else [])) or
                (len(self._target) != len(shape))):
            raise ValueError(f'Input shape {shape} does not match the target shape {self._target} '
                             f"in {'checkpoint' if self._checkpoint else 'layer'} {self._layer} "
                             f'for concatenation over dimension {self._dim}')

    def forward(
            self,
            x: Tensor,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike],
            *_: Any,
            **__: Any) -> Tensor:
        """
        Forward pass of the concatenation layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N,...) and type float, where N is the batch size
        outputs : list[TensorListLike]
            Output from each layer with tensors of shape (N,...) and type float
        checkpoints : list[TensorListLike]
            Output from each checkpoint with tensors of shape (N,...) and type float

        Returns
        -------
        Tensor
            Output tensor of shape (N,...) and type float
        """
        dim: int
        self._check_num_outputs(
            checkpoints[self._layer] if self._checkpoint else outputs[self._layer],
        )

        if self._dim >= 0:
            dim = self._dim + 1
        else:
            dim = self._dim

        if self._checkpoint:
            return torch.cat((x, cast(Tensor, checkpoints[self._layer])), dim=dim)

        return torch.cat((x, cast(Tensor, outputs[self._layer])), dim=dim)

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'{super().extra_repr()}, dim={self._dim}'


class DropPath(BaseLayer):
    """
    Constructs a drop path layer to drop samples in a batch.

    See `FractalNet: Ultra-Deep Neural Networks without Residuals
    <https://arxiv.org/abs/1605.07648>`_ by Larsson et al. (2016) for details.

    Custom version of implementation by `timm
    <https://github.com/pprp/timm/blob/master/timm/layers/drop.py#L157>`_.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the drop path layer
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
        shapes : Shapes | None, default = None
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
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the indexing layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            *args: int,
            map_: bool = True,
            greater: bool = True,
            idx: int | slice | None = None,
            shapes: Shapes | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        idx : int | slice
            Index or slice to apply to the last dimension of the input tensor(s) or if map_ is
            False, the DataList itself
        map_ : bool, default = True
            If slicing should be applied to each tensor in a DataList, or if the DataList should be
            sliced itself, only used if input is a DataList
        shapes : Shapes | None, default = None
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
                'idx instead',
                DeprecationWarning,
                stacklevel=2,
            )
            number = args[0] if args else kwargs.pop('number')
            self._idx = slice(number, None) if greater else slice(number)
        else:
            assert idx is not None
            self._idx = idx

        # If not used as a layer in a network
        if not shapes:
            return

        in_shape = cast(list[int] | list[list[int]], deepcopy(shapes.get(-1, list_=True)))

        # Length of slice
        if isinstance(in_shape[0], int):
            in_shape = self._shape_index(cast(list[int], in_shape))
        elif isinstance(in_shape[0], list) and self._map:
            in_shape = [self._shape_index(shape) for shape in cast(list[list[int]], in_shape)]
        else:
            in_shape = cast(list[list[int]], in_shape)[self._idx]

        shapes.append(in_shape)

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
        x : Tensor
            Input tensor of shape (N,...) and type float, where N is the batch size

        Returns
        -------
        Tensor
            Output tensor of shape (N,...) and type float
        """
        if isinstance(x, DataList) and not self._map:
            return x.get(self._idx, list_=True) if isinstance(self._idx, int) else (
                DataList(x.get(self._idx, list_=True)))
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
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the layer normalisation layer
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
        dims : int | None, default = None
            Number of dimensions to normalise starting with the first dimension, ignoring batch
            dimension, requires shapes argument, won't be used if shape is provided
        shape : list[int] | None, default = None
            Input shape or shape of the first dimension to normalise, will be used if provided, else
            dims will be used
        shapes : Shapes | None, default = None
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
        permutation: ndarray = np.arange(x.ndim - 1, 0, -1)
        return self._layer(x.permute(0, *permutation)).permute(0, *permutation)


class Pack(BaseLayer):
    """
    Constructs a packing layer to combine the output from the previous layer with the output from
    a specified layer into a DataList.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x, outputs, checkpoints) -> DataList[Tensor]
        Forward pass of the pack layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            net_check: bool,
            *args: int | Shapes,
            checkpoint: bool = False,
            layer: int | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        shapes : Shapes
            Shape of the outputs from each layer
        check_shapes : Shapes
            Shape of the outputs from each checkpoint
        checkpoint : bool, default = False
            If layer index should be relative to checkpoint layers
        layer : int | None, default = None
            Layer index to pack with the previous layer, if layer is None, then only the last layer
            will be packed
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        target: list[int] | list[list[int]]
        out_shape: list[list[int]]
        shapes: Shapes = kwargs.pop(
            'shapes',
            None if not args else args[1] if isinstance(args[0], int) else args[0],
        )
        check_shapes: Shapes = kwargs.pop('check_shapes', None if not args else args[-1])
        self._checkpoint: bool = checkpoint or net_check
        self._layer: int | None = layer

        if args and isinstance(args[0], int):
            warn(
                'Passing layer as a positional argument is deprecated, please use layer=',
                DeprecationWarning,
                stacklevel=2,
            )
            self._layer = args[0]

        out_shape = [shapes[-1]] if shapes.check(-1) else shapes.get(-1, True)

        if self._layer is not None:
            target = cast(
                list[int] | list[list[int]],
                (check_shapes if self._checkpoint else shapes).get(self._layer, True),
            )

            if isinstance(target[0], int):
                out_shape.append(cast(list[int], target))
            else:
                out_shape.extend(cast(list[list[int]], target))

        shapes.append(deepcopy(out_shape))

    def forward(
            self,
            x: TensorListLike,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike],
            *_: Any,
            **__: Any) -> DataList[Tensor]:
        """
        Forward pass of the pack layer.

        Parameters
        ----------
        x : TensorListLike
            Input with tensors of shape (N,...) and type float, where N is the batch size
        outputs : list[TensorListLike]
            Output from each layer with tensors of shape (N,...) and type float
        checkpoints : list[TensorListLike]
            Output from each checkpoint with tensors of shape (N,...) and type float

        Returns
        -------
        DataList[Tensor]
            Output DataList with tensors of shape (N,...) and type float
        """
        targets: TensorListLike

        if self._layer is None:
            return DataList([x]) if isinstance(x, Tensor) else x

        targets = (checkpoints if self._checkpoint else outputs)[self._layer]

        if isinstance(x, DataList) and isinstance(targets, DataList):
            x.extend(targets)
        elif isinstance(x, DataList):
            x.append(cast(Tensor, targets))
        elif isinstance(x, Tensor) and isinstance(targets, DataList):
            x = DataList([x])
            x.extend(targets)
        else:
            x = DataList([x, targets])
        return x

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'layer={self._layer}, checkpoint={bool(self._checkpoint)}'


class Reshape(BaseLayer):
    """
    Constructs a reshaping layer to change the data dimensions.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of Reshape
    extra_repr() -> str
        Displays layer parameters when printing the network
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
        factor : bool, default = False
            If reshape should be relative to the network output shape, or if layer is provided,
            which layer to be relative to, requires tracking layer outputs
        layer : int | None, default = None
            If factor is True, which layer for factor to be relative to, if None, network output
            will be used
        net_out : list[int] | None, default = None
            Shape of the network's output, required only if factor is True
        shapes : Shapes | None, default = None
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
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the scale layer
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
        first : bool, default = True
            If dims should count from the first dimension after the batch dimension, or from the
            final dimension backwards
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
            permutation = np.arange(x.ndim - 1, 0, -1)
            return (self._scale * x.permute(0, *permutation)).permute(0, *permutation)
        return self._scale * x

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self._scale.to(*args, **kwargs)
        return self


class Shortcut(BaseMultiLayer):
    """
    Constructs a shortcut layer to add the outputs from two layers.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x, outputs, checkpoints) -> Tensor
        Forward pass of the shortcut layer
    """
    def __init__(
            self,
            net_check: bool,
            layer: int,
            shapes: Shapes,
            check_shapes: Shapes,
            *,
            checkpoint: bool = False,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        layer : int
            Layer index to add the previous layer output with
        shapes : Shapes
            Shape of the outputs from each layer
        check_shapes : Shapes
            Shape of the outputs from each checkpoint
        checkpoint : bool, default = False
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(net_check, layer, shapes, check_shapes, checkpoint=checkpoint, **kwargs)
        idxs: ndarray
        target: ndarray = np.array(self._target)
        shape: ndarray = np.array(shapes[-1].copy())
        mask: ndarray = (shape != 1) & (self._target != 1)
        self._check_addition(shape, mask)

        # If input has any dimensions of length one, output will take the target dimension
        if 1 in shape:
            idxs = np.where(shape == 1)[0]
            shape[idxs] = target[idxs]

        # If target has any dimensions of length one, output will take the input dimension
        if 1 in target:
            idxs = np.where(target == 1)[0]
            shape[idxs] = shape[idxs]

        shapes.append(shape.tolist())

    def _check_addition(self, shape: ndarray, mask: ndarray) -> None:
        """
        Checks if input shape and target shape are compatible for addition.

        Parameters
        ----------
        shape : ndarray
            Input shape
        mask : ndarray
            Mask for dimensions with size of one in both input and target
        """
        if not np.array_equal(shape[mask], np.array(self._target)[mask]):
            raise ValueError(f'Input shape {shape} is not compatible with target shape '
                             f"{self._target} in {'checkpoint' if self._checkpoint else 'layer'} "
                             f'{self._layer} for addition')

    def forward(
            self,
            x: Tensor,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike],
            *_: Any,
            **__: Any) -> Tensor:
        """
        Forward pass of the shortcut layer

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N,...) and type float, where N is the batch size
        outputs : list[TensorListLike]
            Output from each layer with tensors of shape (N,...) and type float
        checkpoints : list[TensorListLike]
            Output from each checkpoint with tensors of shape (N,...) and type float

        Returns
        -------
        Tensor
            Output tensor of shape (N,...) and type float
        """
        self._check_num_outputs(
            checkpoints[self._layer] if self._checkpoint else outputs[self._layer],
        )

        if self._checkpoint:
            return x + cast(Tensor, checkpoints[self._layer])
        return x + cast(Tensor, outputs[self._layer])


class Skip(BaseMultiLayer):
    """
    Bypasses previous layers by retrieving the output from the defined layer.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x, outputs, checkpoints) -> TensorListLike
        Forward pass of the shortcut layer
    """
    def __init__(
            self,
            net_check: bool,
            layer: int,
            shapes: Shapes,
            check_shapes: Shapes,
            *,
            checkpoint: bool = False,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        layer : int
            Layer index to get the output from
        shapes : Shapes
            Shape of the outputs from each layer
        check_shapes : Shapes
            Shape of the outputs from each checkpoint
        checkpoint : bool, default = False
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(net_check, layer, shapes, check_shapes, checkpoint=checkpoint, **kwargs)
        shapes.append(self._target)

    def forward(
            self,
            *_: Any,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike],
            **__: Any) -> TensorListLike:
        """
        Forward pass of the skip layer.

        Parameters
        ----------
        outputs : list[TensorListLike]
            Output from each layer with tensors of shape (N,...) and type float
        checkpoints : list[TensorListLike]
            Output from each checkpoint with tensors of shape (N,...) and type float

        Returns
        -------
        TensorListLike
            Output with tensors of shape (N,...) and type float
        """
        self._check_num_outputs(
            checkpoints[self._layer] if self._checkpoint else outputs[self._layer],
        )
        return cast(Tensor, (checkpoints if self._checkpoint else outputs)[self._layer])


class Unpack(BaseLayer):
    """
    Enables a list of Tensors as input into the network, then selects which Tensor in the list to
    output.

    Methods
    -------
    forward(*_, outputs) -> Tensor
        Forward pass of Unpack
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            net_check: bool,
            index: int,
            shapes: Shapes,
            check_shapes: Shapes,
            *,
            checkpoint: bool = False,
            layer: int = 0,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        index : int
            Index of input Tensor list
        shapes : Shapes
            Shape of the outputs from each layer
        check_shapes : Shapes
            Shape of the outputs from each checkpoint
        checkpoint : bool, default = False
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints
        layer : int, default = 0
            Layer index to get the output from if checkpoint is True
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._checkpoint: bool = checkpoint or net_check
        self._layer: int = layer
        self._idx: int = index
        shapes.append((check_shapes if self._checkpoint else shapes)[(self._layer, self._idx)])

    def forward(
            self,
            *_: Any,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike],
            **__: Any) -> Tensor:
        """
        Forward pass of the skip layer.

        Parameters
        ----------
        outputs : list[TensorListLike]
            Output from each layer with tensors of shape (N,...) and type float, where N is the
            batch size
        checkpoints : list[TensorListLike]
            Output from each checkpoint with tensors of shape (N,...) and type float, where N is the
            batch size

        Returns
        -------
        Tensor
            Output tensor with shape (N,...) and type float
        """
        self._check_num_outputs(
            (checkpoints if self._checkpoint else outputs)[self._layer],
            single=False,
        )
        return cast(Tensor, (checkpoints if self._checkpoint else outputs)[self._layer][self._idx])

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'index={self._idx}, checkpoint={bool(self._checkpoint)}, layer={self._layer}'
