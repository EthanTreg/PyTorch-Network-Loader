"""
Network layers that use the outputs from multiple layers.
"""
from copy import deepcopy
from typing import Any, cast

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray

from netloader.utils.types import TensorListLike
from netloader.utils import Shapes, compare_versions
from netloader.layers.base import BaseMultiLayer, BaseLayer


class Concatenate(BaseMultiLayer):
    """
    Constructs a concatenation layer to combine the outputs from two layers.

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
            net_check: bool,
            layer: int | list[int],
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
        layer : int | list[int]
            Layer index(es) to concatenate the previous layer output with
        shapes : Shapes
            Shape of the outputs from each layer
        check_shapes : Shapes
            Shape of the outputs from each checkpoint
        checkpoint : bool, Optional
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints, default = False
        dim : int, Optional
            Dimension to concatenate to, default = 0
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(net_check, layer, shapes, check_shapes, checkpoint=checkpoint, **kwargs)
        self._dim: int = dim
        target: list[int]
        shape: list[int] = self._shapes[0]

        # If tensors can be concatenated along the specified dimension
        for target in self._shapes[1:]:
            if not self._check_concatenation(shape, target):
                raise ValueError(f'Input shape ({shape}) does not match the target '
                                 f'shape ({target}) from output shapes {self._shapes[1:]} in '
                                 f"{'checkpoints' if self._checkpoint else 'layers'} "
                                 f"{self._layer} for concatenation over dimension {self._dim}")

            shape[self._dim] += target[self._dim]

        shapes.append(shape)

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'dim': self._dim}

    def _check_concatenation(self, shape1: list[int], shape2: list[int]) -> bool:
        """
        Checks if input shape and target shape are compatible for concatenation.

        Parameters
        ----------
        shape1 : list[int]
            First shape
        shape2 : list[int]
            Second shape

        Returns
        -------
        bool
            Whether the shapes are compatible for concatenation
        """
        if ((shape2[:self._dim] + (shape2[self._dim + 1:] if self._dim != -1 else []) !=
             shape1[:self._dim] + (shape1[self._dim + 1:] if self._dim != -1 else [])) or
                (len(shape2) != len(shape1))):
            return False
        return True

    def forward(
            self,
            x: TensorListLike,
            *_: Any,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike],
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
        return torch.cat(
            self._forward(x, outputs, checkpoints).get_data(False),
            dim=self._dim + 1 if self._dim >= 0 else self._dim,
        )

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'{super().extra_repr()}, dim={self._dim}'


class Pack(BaseMultiLayer):
    """
    Constructs a packing layer to combine the output from the previous layer with the output from
    a specified layer into a DataList.

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
            net_check: bool,
            shapes: Shapes,
            check_shapes: Shapes,
            *,
            checkpoint: bool = False,
            layer: int | list[int] | None = None,
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
        checkpoint : bool, Optional
            If layer index should be relative to checkpoint layers, default = False
        layer : int | None, Optional
            Layer index to pack with the previous layer, if layer is None, then only the last layer
            will be packed
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            net_check,
            layer or 0,
            shapes,
            check_shapes,
            checkpoint=checkpoint,
            **kwargs,
        )

        if layer is None:
            self._layer = []
            self._shapes = self._shape_flatten(deepcopy(shapes.get(-1, True)))

        shapes.append(self._shapes)


class Shortcut(BaseMultiLayer):
    """
    Constructs a shortcut layer to add the outputs from two layers.

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
            net_check: bool,
            layer: int | list[int],
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
        layer : int | list[int]
            Layer index(es) to add the previous layer output with
        shapes : Shapes
            Shape of the outputs from each layer
        check_shapes : Shapes
            Shape of the outputs from each checkpoint
        checkpoint : bool, Optional
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints, default = False
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(net_check, layer, shapes, check_shapes, checkpoint=checkpoint, **kwargs)
        idxs: ndarray
        mask: ndarray
        shapes_: ndarray = np.array(self._shapes)
        shape: ndarray = shapes_[0]

        mask = (shapes_[0] != 1) & (shapes_[1:] != 1)
        self._check_addition(shapes_[0], shapes_[1:], mask)

        # If input has any dimensions of length one, output will take the target dimension
        if 1 in shape:
            idxs = np.where(shape == 1)[0]
            shape[idxs] = np.max(shapes_[:, idxs], axis=0)

        shapes.append(shape.tolist())

    def _check_addition(self, shape: ndarray, target: ndarray, mask: ndarray) -> None:
        """
        Checks if input shape and target shape are compatible for addition.

        Parameters
        ----------
        shape : ndarray
            Input shape
        mask : ndarray
            Mask for dimensions with size of one in both input and target
        """
        if np.ndim(target) == 2 and all(
                np.array_equal(shape[mask[i]], target[i, mask[i]]) for i in range(len(target))):
            return
        if np.ndim(target) == 1 and np.array_equal(shape[mask], target[mask]):
            return
        raise ValueError(f'Input shape {shape} is not compatible with target shape {target} in '
                         f"{'checkpoint' if self._checkpoint else 'layer'} "
                         f'{self._layer} for addition')

    def forward(
            self,
            x: TensorListLike,
            *_: Any,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike],
            **__: Any) -> Tensor:
        """
        Forward pass of the shortcut layer.

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
        return torch.sum(torch.stack(self._forward(x, outputs, checkpoints).get_data(False)), dim=0)


class Skip(BaseMultiLayer):
    """
    Bypasses previous layers by retrieving the output from the defined layer.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    """
    _layer: int

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
        checkpoint : bool, Optional
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints, default = False
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        assert isinstance(layer, int)
        super().__init__(net_check, layer, shapes, check_shapes, checkpoint=checkpoint, **kwargs)
        shapes.append(cast(list[int], self._target))

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
        return (checkpoints if self._checkpoint else outputs)[self._layer]


class Unpack(BaseLayer):
    """
    Enables a list of Tensors as input into the network, then selects which Tensor in the list to
    output.
    """
    def __init__(
            self,
            net_check: bool,
            index: int,
            shapes: Shapes,
            check_shapes: Shapes,
            *,
            checkpoint: bool = False,
            layer: int | None = None,
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
        checkpoint : bool, Optional
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints, default = False
        layer : int | None, Optional
            Layer index to get the output from if checkpoint is True, default = None
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._checkpoint: bool = checkpoint or net_check
        self._layer: int | None = layer \
            if layer or compare_versions(self._ver, '3.11.0') else 0
        self._idx: int = index

        if self._layer is None:
            shapes.append(shapes[(-1, self._idx)].copy())
        else:
            shapes.append(
                (check_shapes if self._checkpoint else shapes)[(self._layer, self._idx)].copy(),
            )

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'checkpoint': self._checkpoint,
            'layer': self._layer,
            'index': self._idx,
        }

    def forward(
            self,
            x: TensorListLike,
            *_: Any,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike],
            **__: Any) -> Tensor:
        """
        Forward pass of the unpack layer.

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
        if self._layer is None:
            x = self._check_num_outputs(
                x,
                single=False,
            )
            return x.get(self._idx, True)

        inputs: TensorListLike = (checkpoints if self._checkpoint else outputs)[self._layer]
        inputs = self._check_num_outputs(
            inputs,
            single=False,
        )
        return inputs.get(self._idx, True)

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'index={self._idx}, checkpoint={bool(self._checkpoint)}, layer={self._layer}'


__all__ = ['Concatenate', 'Pack', 'Shortcut', 'Skip', 'Unpack']
