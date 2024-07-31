"""
Layer utility functions
"""
from typing import Any, Self

import torch
import numpy as np
from torch import nn, Tensor

from netloader.utils.utils import check_params


class BaseLayer(nn.Module):
    """
    Base layer for other layers to inherit

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers part of the parent layer to loop through in the forward pass

    Methods
    -------
    initialise_layers()
        Checks if self.layers contains layers and exposes them to the optimiser so that weights can
        be trained
    forward(x, **_) -> Tensor
        Forward pass for a generic layer
    """
    def __init__(self, idx: int, **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number

        **kwargs
            Leftover parameters for checking if they are valid
        """
        super().__init__()
        self._device: torch.device = torch.device('cpu')
        self.layers: list[nn.Module] | nn.Sequential = []
        supported_params: list[str] = [
            'net_check',
            'net_out',
            'shapes',
            'check_shapes',
            'type',
            'group',
        ]

        if kwargs:
            check_params(
                f'{self.__class__.__name__} in layer {idx}',
                supported_params,
                np.array(list(kwargs.keys())),
            )

    @staticmethod
    def _check_options(name: str, value: str | None, options: set[str | None]) -> None:
        """
        Checks if a provided option is supported

        Parameters
        ----------
        name : str
            Name of the option
        value : str
            Value provided
        options : set[str]
            List of supported options
        """
        if value not in options:
            raise ValueError(f'{name.title()} ({value}) is unknown, {name} must be one of '
                             f'{options}')

    @staticmethod
    def _check_stride(stride: int | list[int]) -> None:
        """
        Checks if the stride is greater than 1 if 'same' padding is being used

        Parameters
        ----------
        stride : int | list[int]
            Stride of the kernel
        """
        if (np.array(stride) > 1).any():
            raise ValueError(f"'same' padding is not supported for strides > 1 ({stride})")

    @staticmethod
    def _check_shape(shape: list[int]) -> None:
        """
        Checks if the input shape has more than 4 dimensions or fewer than 2

        Parameters
        ----------
        shape : list[int]
            Input shape
        """
        if not 1 < len(shape) < 5:
            raise ValueError(f'Tensors with more than 4 dimensions or less than 2 is not '
                             f'supported, input shape is {shape}')

    @staticmethod
    def _check_factor_filters(
            shape: list[int],
            filters: int | None = None,
            factor: float | None = None,
            target: list[int] | None = None) -> list[int]:
        """
        Checks if factor or filters is provided and calculates the number of filters

        Parameters
        ----------
        shape : list[int]
            Input shape
        filters : int, optional
            Number of convolutional filters, will be used if provided, else factor will be used
        factor : float, optional
            Number of convolutional filters equal to the output channels multiplied by factor,
            won't be used if filters is provided
        target : list[int]
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

    def initialise_layers(self) -> None:
        """
        Checks if self.layers contains layers and exposes them to the optimiser so that weights can
        be trained
        """
        if isinstance(self.layers, list) and self.layers:
            self.layers = nn.Sequential(*self.layers)

    def forward(self, x: Tensor, **_: Any) -> Tensor:
        """
        Forward pass for a generic layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor with batch size N

        Returns
        -------
        (N,...) Tensor
            Output tensor with batch size N
        """
        assert isinstance(self.layers, nn.Sequential)
        return self.layers(x)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self._device, *_ = torch._C._nn._parse_to(*args, **kwargs)
        return self


class BaseMultiLayer(BaseLayer):
    """
    Base layer for layers that use earlier layers to inherit

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers part of the parent layer to loop through in the forward pass

    Methods
    -------
    extra_repr() -> string
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            net_check: bool,
            layer: int,
            shapes: list[list[int]],
            check_shapes: list[list[int]],
            checkpoint: bool = False,
            **kwargs: Any):
        """
        Parameters
        ----------
        net_check : boolean
            If layer index should be relative to checkpoint layers
        layer : integer
            Layer index to concatenate the previous layer output with
        shapes : list[list[integer]]
            Shape of the outputs from each layer
        check_shapes : list[list[integer]]
            Shape of the outputs from each checkpoint
        checkpoint : boolean, default = False
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
        Displays layer parameters when printing the network

        Returns
        -------
        string
            Layer parameters
        """
        return f'layer={self._layer}, checkpoint={bool(self._checkpoint)}'


def _int_list_conversion(length: int, elements: list[int | list[int]]) -> list[list[int]]:
    """
    Converts integers to a list of integers, if integer is already a list of integers, then list
    will be preserved

    Parameters
    ----------
    length : int
        Number of elements in the converted list of integers
    elements : list[int | list[int]]
        Integers to convert to list of integers, or list of integers to remain unchanged

    Returns
    -------
    list[list[int]]
        List of integers for each inputted integer or list of integers
    """
    lists: list[list[int]] = []

    for element in elements:
        if isinstance(element, int):
            lists.append([element] * length)
        else:
            lists.append(element)

    return lists


def _kernel_shape(
        kernel: int | list[int],
        strides: int | list[int],
        padding: int | list[int],
        shape: list[int]) -> list[int]:
    """
    Calculates the output shape after a kernel operation

    Parameters
    ----------
    kernel : int | list[int]
        Size of the kernel
    strides : int | list[int]
        Stride of the kernel
    padding : int | list[int]
        Input padding
    shape : list[int]
        Input shape of the layer

    Returns
    -------
    list[int]
        Output shape of the layer
    """
    shape = shape.copy()
    strides, kernel, padding = _int_list_conversion(
        len(shape[1:]),
        [strides, kernel, padding],
    )

    for i, (stride, kernel_length, pad, length) in enumerate(zip(
            strides,
            kernel,
            padding,
            shape[1:]
    )):
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
        Input shape of the layer
    out_shape : list[int]
        Output shape from the layer

    Returns
    -------
    list[int]
        Required padding for specific output shape
    """
    padding: list[int] = []
    strides, kernel = _int_list_conversion(len(in_shape[1:]), [strides, kernel])

    for stride, kernel_length, in_length, out_length in zip(
            strides,
            kernel,
            in_shape[1:],
            out_shape[1:],
    ):
        padding.append((stride * (out_length - 1) + kernel_length - in_length) // 2)

    return padding
