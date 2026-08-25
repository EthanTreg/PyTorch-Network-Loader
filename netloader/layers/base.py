"""
Base classes for network layers
"""
import logging as log
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Any, Self, Literal, Sequence, cast, overload

import torch
import numpy as np
from torch import nn, Tensor

import netloader
from netloader.data import DataList
from netloader.utils import Shapes, check_params
from netloader.utils.types import TensorListLike, ShapeLike, ShapeListLike


class BaseLayer(nn.Module, ABC):
    """
    Base layer for other layers to inherit.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    """
    _ver: str
    _device: torch.device = torch.device('cpu')
    _logger: log.Logger = log.getLogger(__name__)
    group: int
    description: str

    def __init__(
            self,
            *,
            idx: int = 0,
            group: int = 0,
            version: str = '',
            description: str = '',
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        idx : int, Optional
            Layer number, default = 0
        group : int, Optional
            Which group the layer belongs to, if 0 it will always be used, else it will only be used
            if the Network group matches the layer's group, default = 0
        version : str, Optional
            Version of the network, default = netloader.__version__
        description : str, Optional
            Description of the layer
        **kwargs
            Leftover parameters for checking if they are valid
        """
        super().__init__()
        supported_params: list[str] = [
            'type',
            'root',
            'net_check',
            'net_out',
            'shapes',
            'check_shapes',
        ]
        self._ver = version or netloader.__version__
        self.group = group
        self.description = description

        if kwargs:
            check_params(
                f'{self.__class__.__name__} in layer {idx}',
                supported_params,
                np.array(list(kwargs.keys())),
            )

    def __getstate__(self) -> dict[str, Any]:
        """
        Gets the state of the layer for saving.

        Returns
        -------
        dict[str, Any]
            State of the layer
        """
        return {
            'type': self.__class__.__name__,
            'version': self._ver,
            'group': self.group,
            'description': self.description,
        }

    @staticmethod
    def _check_options(name: str, value: str | None, options: set[str | None]) -> None:
        """
        Checks if a provided option is supported.

        Parameters
        ----------
        name : str
            Name of the option
        value : str | None
            Value provided
        options : set[str | None]
            List of supported options
        """
        if value not in options:
            raise ValueError(f'{name.title()} ({value}) is unknown, {name} must be one of '
                             f'{options}')

    @staticmethod
    def _check_stride(stride: int | list[int]) -> None:
        """
        Checks if the stride is greater than 1 if 'same' padding is being used.

        Parameters
        ----------
        stride : int | list[int]
            Stride of the kernel
        """
        if (np.array(stride) > 1).any():
            raise ValueError(f"'same' padding is not supported for strides > 1 ({stride})")

    @staticmethod
    def _check_factor_filters(
            shape: list[int],
            *,
            filters: int | None = None,
            factor: float | None = None,
            target: list[int] | None = None) -> list[int]:
        """
        Checks if factor or filters is provided and calculates the number of filters.

        Parameters
        ----------
        shape : list[int]
            Input shape
        filters : int | None, Optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float | None, Optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        target : list[int] | None, Optional
            Target shape that factor is relative to, required only if layer contains factor

        Returns
        -------
        list[int]
            Input shape with the adjusted number of filters
        """
        if factor is not None and target is not None:
            shape[0] = max(1, int(target[0] * factor))
        elif filters is not None:
            shape[0] = filters
        else:
            raise ValueError('Either factor or filters is required')
        return shape

    @staticmethod
    def _check_shape(dims: tuple[int, int], shape: list[int]) -> None:
        """
        Checks if the input shape has more than 4 dimensions or fewer than 2.

        Parameters
        ----------
        shape : list[int]
            Input shape
        """
        if not dims[0] <= len(shape) <= dims[1]:
            raise ValueError(f'Tensors with more than {dims[1]} dimensions or less than {dims[0]} '
                             f'is not supported, input shape is {shape}')

    @overload
    @staticmethod
    def _check_num_outputs(output: TensorListLike, single: Literal[True]) -> Tensor: ...

    @overload
    @staticmethod
    def _check_num_outputs(output: TensorListLike, single: Literal[False]) -> DataList[Tensor]: ...

    @staticmethod
    def _check_num_outputs(output: TensorListLike, single: bool = True) -> TensorListLike:
        """
        Checks if the output is a single tensor or a list of tensors.

        Parameters
        ----------
        output : TensorListLike
            Output from the layer with tensors of shape (N,...) and type float, where N is the
            batch size
        single : bool, Optional
            If the output should be a single tensor, if False it will be a list of tensors,
            default = True

        Returns
        -------
        TensorListLike
            Output from the layer unchanged with tensors of shape (N,...) and type float
        """
        if single and isinstance(output, DataList):
            raise ValueError(f'Output must be a single tensor, but {len(output)} tensors were '
                             f'provided, use an Unpack layer first')
        if not single and isinstance(output, Tensor):
            raise ValueError('Output must be a list of tensors, but a single tensor was '
                             'provided')
        return output

    @abstractmethod
    def forward(self, x: Any, *_: Any, **__: Any) -> Any:
        """
        Forward pass for child layer ti implement.

        Parameters
        ----------
        x : Any
            Layer input

        Returns
        -------
        Any
            Layer output
        """

    def clear_checkpoints(self) -> None:
        """
        Clears the checkpoints for the layer.
        """

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self._device, *_ = torch._C._nn._parse_to(*args, **kwargs)  # pylint: disable=protected-access
        return self


class BaseSingleLayer(BaseLayer):
    """
    Base layer for layers that only use the previous layer.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Network's
    description : str
        Description of the layer
    layers : Sequential
        Layers to loop through in the forward pass
    """
    layers: nn.Sequential

    def __init__(self, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self.layers = nn.Sequential()

    def forward(self, x: Any, *_: Any, **__: Any) -> Any:
        """
        Forward pass for a generic layer.

        Parameters
        ----------
        x : TensorListLike
            Input with tensors of shape (N,...) and type float, where N is the batch size

        Returns
        -------
        TensorListLike
            Output with tensors of shape (N,...) and type float
        """
        return self.layers(x)


class BaseMultiLayer(BaseLayer):
    """
    Base layer for layers that use earlier layers to inherit.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    """
    _checkpoint: bool
    _layer: int | list[int]
    _shapes: list[list[int]]
    _target: ShapeLike

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
            Layer index(es) to concatenate the previous layer output with
        shapes : Shapes
            Shape of the outputs from each layer
        check_shapes : Shapes
            Shape of the outputs from each checkpoint
        checkpoint : bool, Optional
            If layer index should be relative to checkpoint layers, default = False
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._checkpoint = checkpoint or net_check
        self._layer = layer
        self._shapes = []
        i: int
        idx: int = 1 if shapes.check(-1) else len(shapes.get(-1, True))
        shapes_: Shapes = check_shapes if self._checkpoint else shapes

        self._shapes = self._shape_flatten([
            deepcopy(shapes.get(-1, True)),
            [deepcopy(shapes_.get(i, True))
             for i in ([self._layer] if isinstance(self._layer, int) else self._layer)],
        ])
        self._target = self._shapes[idx] \
            if isinstance(self._layer, int) and len(self._shapes) == idx + 1 else self._shapes[idx:]


    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'checkpoint': self._checkpoint, 'layer': self._layer}

    @staticmethod
    def _shape_flatten(shapes: ShapeListLike) -> list[list[int]]:
        """
        Flattens a list of layer output shapes into a single list of array output shapes.

        Parameters
        ----------
        shapes : ShapeLike | list[ShapeLike]
            Layer output shape or list of layer output shapes to flatten

        Returns
        -------
        list[list[int]]
            Flattened list of array output shapes
        """
        new_shapes: list[list[int]] = []
        shape: ShapeListLike

        if isinstance(shapes[0], int):
            return [cast(list[int], shapes)]

        for shape in cast(Sequence[ShapeListLike], shapes):
            if isinstance(shape[0], int):
                new_shapes.append(cast(list[int], shape))
            else:
                new_shapes.extend(BaseMultiLayer._shape_flatten(shape))
        return new_shapes

    def _forward(
            self,
            x: TensorListLike,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike]) -> DataList[Tensor]:
        """
        Forward pass to pack multi-layer data into a DataList.

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
            Output with tensors of shape (N,...) and type float
        """
        outputs_: list[TensorListLike] = checkpoints if self._checkpoint else outputs
        return DataList(sum((
            [cast(Tensor, outputs_[i])] if isinstance(outputs_[i], Tensor) else list(outputs_[i])
            for i in ([self._layer] if isinstance(self._layer, int) else self._layer)
        ), start=[x] if isinstance(x, Tensor) else list(x)))

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'layer={self._layer}, checkpoint={bool(self._checkpoint)}'

    def forward(
            self,
            x: TensorListLike,
            *_: Any,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike],
            **__: Any) -> TensorListLike:
        """
        Forward pass to combine outputs from multiple layers.

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
        TensorListLike
            Output with tensors of shape (N,...) and type float
        """
        return self._forward(x, outputs=outputs, checkpoints=checkpoints)


__all__ = ['BaseLayer', 'BaseSingleLayer', 'BaseMultiLayer']
