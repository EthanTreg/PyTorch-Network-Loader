"""
Base classes for network layers
"""
import logging as log
from warnings import warn
from typing import Any, Self

import torch
import numpy as np
from torch import nn, Tensor

from netloader.data import DataList
from netloader.utils.types import TensorListLike
from netloader.utils import Shapes, check_params


class DeprecatedMetaclass(type):
    """
    Metaclass to warn about deprecated attributes in BaseLayer.
    """
    def __getattribute__(cls, item: str) -> Any:
        """
        Raises deprecation warnings for deprecated attributes.

        Parameters
        ----------
        item : str
            Attribute name

        Returns
        -------
        Any
            Attribute value
        """
        if item == 'layers':
            warn(
                'layers attribute is deprecated is BaseLayer, please inherit '
                'BaseSingleLayer',
                DeprecationWarning,
                stacklevel=2,
            )
        return type.__getattribute__(cls, item)


class BaseLayer(nn.Module, metaclass=DeprecatedMetaclass):
    """
    Base layer for other layers to inherit.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    initialise_layers()
        Checks if self.layers contains layers and exposes them to the optimiser so that weights can
        be trained
    forward(x) -> Tensor
        Forward pass for a generic layer
    """
    def __init__(self, *, idx: int = 0, group: int = 0, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        idx : int, default = 0
            Layer number
        group : int, default = 0
            Which group the layer belongs to, if 0 it will always be used, else it will only be used
            if the Network group matches the layer's group
        **kwargs
            Leftover parameters for checking if they are valid
        """
        super().__init__()
        supported_params: list[str] = [
            'net_check',
            'net_out',
            'shapes',
            'check_shapes',
            'type',
        ]
        self.group: int = group
        self.layers: nn.Sequential = nn.Sequential()
        self._device: torch.device = torch.device('cpu')
        self._logger: log.Logger = log.getLogger(__name__)

        if kwargs:
            check_params(
                f'{self.__class__.__name__} in layer {idx}',
                supported_params,
                np.array(list(kwargs.keys())),
            )

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
        filters : int | None, default = None
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float | None, default = None
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        target : list[int] | None, default = None
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
    def _check_shape(shape: list[int]) -> None:
        """
        Checks if the input shape has more than 4 dimensions or fewer than 2.

        Parameters
        ----------
        shape : list[int]
            Input shape
        """
        if not 1 < len(shape) < 5:
            raise ValueError(f'Tensors with more than 4 dimensions or less than 2 is not '
                             f'supported, input shape is {shape}')

    @staticmethod
    def _check_num_outputs(output: TensorListLike, single: bool = True) -> None:
        """
        Checks if the output is a single tensor or a list of tensors.

        Parameters
        ----------
        output : TensorListLike
            Output from the layer with tensors of shape (N,...) and type float, where N is the
            batch size
        single : bool, default = True
            If the output should be a single tensor, if False it will be a list of tensors
        """
        if single and isinstance(output, DataList):
            raise ValueError(f'Output must be a single tensor, but {len(output)} tensors were '
                             f'provided, use an Unpack layer first')
        if not single and isinstance(output, Tensor):
            raise ValueError('Output must be a list of tensors, but a single tensor was '
                             'provided')

    def initialise_layers(self) -> None:
        """
        Checks if self.layers contains layers and exposes them to the optimiser so that weights can
        be trained.

        This is for backwards compatibility with layers being provided as lists, which is
        deprecated.
        """
        if isinstance(self.layers, list) and self.layers:  # type: ignore[unreachable]
            warn(  # type: ignore[unreachable]
                'Layers of type list is deprecated, please use nn.Sequential',
                DeprecationWarning,
                stacklevel=2,
            )
            self.layers = nn.Sequential(*self.layers)

    def forward(self, x: Any, *_: Any, **__: Any) -> Any:
        """
        Forward pass for a generic layer.

        Parameters
        ----------
        x : Tensor
            Input with tensors of shape (N,...) and type float, where N is the batch size

        Returns
        -------
        Tensor
            Output with tensors of shape (N,...) and type float
        """
        warn(
            'Using forward pass from BaseLayer is deprecated, please inherit '
            'BaseSingleLayer or implement a custom forward method',
            DeprecationWarning,
            stacklevel=2,
        )
        return self.layers(x)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self._device, *_ = torch._C._nn._parse_to(*args, **kwargs)  # pylint: disable=protected-access
        return self


class BaseSingleLayer(BaseLayer, metaclass=type):
    """
    Base layer for layers that only use the previous layer.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : nn.Sequential
        Layers part of the parent layer to loop through in the forward pass

    Methods
    -------
    forward(x, outputs, checkpoints) -> Tensor
        Forward pass for a generic layer
    """
    def forward(self, x: Tensor, *_: Any, **__: Any) -> Tensor:
        """
        Forward pass for a generic layer.

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (N,...) and type float, where N is the batch size

        Returns
        -------
        Tensor
            Output tensor with shape (N,...) and type float
        """
        return self.layers(x)


class BaseMultiLayer(BaseLayer):
    """
    Base layer for layers that use earlier layers to inherit.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    extra_repr() -> st
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
            If layer index should be relative to checkpoint layers
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._checkpoint: bool = checkpoint or net_check
        self._layer: int = layer
        self._target: list[int]

        # If checkpoints are being used
        if self._checkpoint:
            self._target = check_shapes[self._layer].copy()
        else:
            self._target = shapes[self._layer].copy()

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'layer={self._layer}, checkpoint={bool(self._checkpoint)}'
