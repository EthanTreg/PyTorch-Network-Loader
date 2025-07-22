"""
Miscellaneous network layers
"""
from typing import Any, Self

import torch
import numpy as np
from torch import nn, Tensor
from numpy import ndarray

from netloader.layers.base import BaseLayer, BaseMultiLayer


class Checkpoint(BaseLayer):
    """
    Constructs a layer to create a checkpoint for using the output from the previous layer in
    future layers

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, checkpoints, **_) -> Tensor
        Forward pass of the checkpoint layer
    """
    def __init__(
            self,
            shapes: list[list[int]],
            check_shapes: list[list[int]],
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        shapes : list[list[int]]
            Shape of the outputs from each layer
        check_shapes : list[list[int]]
            Shape of the outputs from each checkpoint

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._check_num: int = len(check_shapes)
        shapes.append(shapes[-1].copy())
        check_shapes.append(shapes[-1].copy())

    def forward(self, x: Tensor, checkpoints: list[Tensor], *_: Any, **__: Any) -> Tensor:
        """
        Forward pass of the checkpoint layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor
        checkpoints : list[Tensor]
            List of checkpoint values

        Returns
        -------
        (N,...) Tensor
            Output tensor
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
    Constructs a concatenation layer to combine the outputs from two layers

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, outputs, checkpoints, **_) -> Tensor
        Forward pass of the concatenation layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            net_check: bool,
            layer: int,
            shapes: list[list[int]],
            check_shapes: list[list[int]],
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
        shapes : list[list[int]]
            Shape of the outputs from each layer
        check_shapes : list[list[int]]
            Shape of the outputs from each checkpoint
        checkpoint : bool, default = False
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints
        dim : int, default = 0
            Dimension to concatenate to

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            net_check,
            layer,
            shapes,
            check_shapes,
            checkpoint=checkpoint,
            **kwargs,
        )
        self._dim: int = dim
        shape: list[int] = shapes[-1].copy()

        # If tensors cannot be concatenated along the specified dimension
        self._check_concatenation(shape)

        shape[self._dim] = shape[self._dim] + self._target[self._dim]
        shapes.append(shape)

    def _check_concatenation(self, shape: list[int]) -> None:
        """
        Checks if input shape and target shape are compatible for concatenation

        Parameters
        ----------
        shape : list[int]
            Input shape
        """
        if ((self._target[:self._dim] + self._target[self._dim + 1:] !=
             shape[:self._dim] + shape[self._dim + 1:]) or
                (len(self._target) != len(shape))):
            raise ValueError(f'Input shape {shape} does not match the target shape {self._target} '
                             f"in {'checkpoint' if self._checkpoint else 'layer'} {self._layer} "
                             f'for concatenation over dimension {self._dim}')

    def forward(
            self,
            x: Tensor,
            outputs: list[Tensor],
            checkpoints: list[Tensor],
            *_: Any,
            **__: Any) -> Tensor:
        """
        Forward pass of the concatenation layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor
        outputs : list[Tensor]
            Output from each layer
        checkpoints : list[Tensor]
            Output from each checkpoint

        Returns
        -------
        (N,...) Tensor
            Output tensor
        """
        dim: int

        if self._dim >= 0:
            dim = self._dim + 1
        else:
            dim = self._dim

        if self._checkpoint:
            return torch.cat((x, checkpoints[self._layer]), dim=dim)

        return torch.cat((x, outputs[self._layer]), dim=dim)

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'{super().extra_repr()}, dim={self._dim}'


class DropPath(BaseLayer):
    """
    Constructs a drop path layer to drop samples in a batch

    See `FractalNet: Ultra-Deep Neural Networks without Residuals
    <https://arxiv.org/abs/1605.07648>`_ by Larsson et al. (2016) for details.

    Custom version of implementation by `timm
    <https://github.com/pprp/timm/blob/master/timm/layers/drop.py#L157>`_.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, **_) -> Tensor
        Forward pass of the drop path layer
    """
    def __init__(self, prob: float, shapes: list[list[int]] | None = None, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        prob : float
            Probability of dropout
        shapes : list[list[int]], optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**{'idx': 0} | kwargs)
        self._keep_prob: float = 1 - prob

        # If not used as a layer in a network
        if not shapes:
            return

        shapes.append(shapes[-1].copy())

    def forward(self, x: Tensor, *_: Any, **__: Any) -> Tensor:
        """
        Forward pass of the drop path layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor

        Returns
        -------
        (N,...) Tensor
            Output tensor
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
    Constructs a layer to slice the last dimension from the output from the previous layer

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, **_) -> Tensor
        Forward pass of the indexing layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            number: int,
            shapes: list[list[int]] | None = None,
            greater: bool = True,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        number : int
            Number of values to slice, can be negative
        shapes : list[list[int]], optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary
        greater : bool, default = True
            If slicing should include all values greater or less than number index

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**{'idx': 0} | kwargs)
        self._greater: bool = greater
        self._number: int = number

        # If not used as a layer in a network
        if not shapes:
            return

        shapes.append(shapes[-1].copy())

        # Length of slice
        if (self._greater and self._number < 0) or (not self._greater and self._number > 0):
            shapes[-1][-1] = abs(self._number)
        else:
            shapes[-1][-1] = shapes[-1][-1] - abs(number)

    def forward(self, x: Tensor, *_: Any, **__: Any) -> Tensor:
        """
        Forward pass of the indexing layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor

        Returns
        -------
        (N,...) Tensor
            Output tensor
        """
        if self._greater:
            return x[..., self._number:]

        return x[..., :self._number]

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'greater={bool(self._greater)}, number={self._number}'


class LayerNorm(BaseLayer):
    """
    Constructs a layer normalisation layer that is the same as PyTorch's LayerNorm, except
    normalisation priority is the first dimension after batch dimension to follow ConvNeXt
    implementation

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, **_) -> Tensor
        Forward pass of the layer normalisation layer
    """
    def __init__(
            self,
            dims: int | None = None,
            shape: list[int] | None = None,
            shapes: list[list[int]] | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        dims : int, optional
            Number of dimensions to normalise starting with the first dimension, ignoring batch
            dimension, requires shapes argument, won't be used if shape is provided
        shape : list[int], optional
            Input shape or shape of the first dimension to normalise, will be used if provided, else
            dims will be used
        shapes : list[list[int]], optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**{'idx': 0} | kwargs)
        self._layer: nn.LayerNorm

        if not shape and dims and shapes:
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
        Forward pass of the layer normalisation layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor
        _

        Returns
        -------
        (N,...) Tensor
            Output Tensor
        """
        permutation: ndarray = np.arange(x.ndim - 1, 0, -1)
        return self._layer(x.permute(0, *permutation)).permute(0, *permutation)


class Reshape(BaseLayer):
    """
    Constructs a reshaping layer to change the data dimensions

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, **_) -> Tensor
        Forward pass of Reshape
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            shape: list[int],
            layer: int | None = None,
            net_out: list[int] | None = None,
            shapes: list[list[int]] | None = None,
            factor: bool = False,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        shape : list[int]
            Desired shape of the output tensor, ignoring first dimension
        layer : int, optional
            If factor is True, which layer for factor to be relative to, if None, network output
            will be used
        net_out : list[int], optional
            Shape of the network's output, required only if factor is True
        shapes : list[list[int]], optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary
        factor : bool, default = False
            If reshape should be relative to the network output shape, or if layer is provided,
            which layer to be relative to, requires tracking layer outputs

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**{'idx': 0} | kwargs)
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
        Checks if the input tensor can be reshaped into the output tensor

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
        Forward pass of reshaping tensors

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor

        Returns
        -------
        (N,...) Tensor
            Output tensor
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
    Constructs a scaling layer that scales the output by a learnable tensor

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, **_) -> Tensor
        Forward pass of the scale layer
    """
    def __init__(
            self,
            dims: int,
            scale: float,
            shapes: list[list[int]],
            first: bool = True,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        dims : int
            Number of dimensions to have individual scales for
        scale : float
            Initial scale factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        first : bool, default = True
            If dims should count from the first dimension after the batch dimension, or from the
            final dimension backwards

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**{'idx': 0} | kwargs)
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
        Forward pass of the scale layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor

        Returns
        -------
        (N,...) Tensor
            Output tensor
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
    Constructs a shortcut layer to add the outputs from two layers

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, outputs, checkpoints, **_) -> Tensor
        Forward pass of the shortcut layer
    """
    def __init__(
            self,
            net_check: bool,
            layer: int,
            shapes: list[list[int]],
            check_shapes: list[list[int]],
            checkpoint: bool = False,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        layer : int
            Layer index to add the previous layer output with
        shapes : list[list[int]]
            Shape of the outputs from each layer
        check_shapes : list[list[int]]
            Shape of the outputs from each checkpoint
        checkpoint : bool, default = False
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            net_check,
            layer,
            shapes,
            check_shapes,
            checkpoint=checkpoint,
            **kwargs,
        )
        idxs: ndarray
        shape: ndarray = np.array(shapes[-1].copy())
        mask: ndarray = (shape != 1) & (self._target != 1)
        self._target = np.array(self._target)
        self._check_addition(shape, mask)

        # If input has any dimensions of length one, output will take the target dimension
        if 1 in shape:
            idxs = np.where(shape == 1)[0]
            shape[idxs] = self._target[idxs]

        # If target has any dimensions of length one, output will take the input dimension
        if 1 in self._target:
            idxs = np.where(self._target == 1)[0]
            shape[idxs] = shape[idxs]

        shapes.append(shape.tolist())

    def _check_addition(self, shape: ndarray, mask: ndarray) -> None:
        """
        Checks if input shape and target shape are compatible for addition

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
            outputs: list[Tensor],
            checkpoints: list[Tensor],
            *_: Any,
            **__: Any) -> Tensor:
        """
        Forward pass of the shortcut layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor
        outputs : list[Tensor]
            Output from each layer
        checkpoints : list[Tensor]
            Output from each checkpoint

        Returns
        -------
        (N,...) Tensor
            Output tensor
        """
        if self._checkpoint:
            return x + checkpoints[self._layer]

        return x + outputs[self._layer]


class Skip(BaseMultiLayer):
    """
    Bypasses previous layers by retrieving the output from the defined layer

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, outputs, checkpoints, **_) -> Tensor
        Forward pass of the shortcut layer
    """
    def __init__(
            self,
            net_check: bool,
            layer: int,
            shapes: list[list[int]],
            check_shapes: list[list[int]],
            checkpoint: bool = False,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        layer : int
            Layer index to get the output from
        shapes : list[list[int]]
            Shape of the outputs from each layer
        check_shapes : list[list[int]]
            Shape of the outputs from each checkpoint
        checkpoint : bool, default = False
            If layer index should be relative to checkpoint layers or network layers, if checkpoints
            in net is True, layer will always be relative to checkpoints

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            net_check,
            layer,
            shapes,
            check_shapes,
            checkpoint=checkpoint,
            **kwargs,
        )
        shapes.append(self._target)

    def forward(
            self,
            x: Tensor,
            outputs: list[Tensor],
            checkpoints: list[Tensor],
            *_: Any,
            **__: Any) -> Tensor:
        """
        Forward pass of the skip layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor
        outputs : list[Tensor]
            Output from each layer
        checkpoints : list[Tensor]
            Output from each checkpoint

        Returns
        -------
        (N,...) Tensor
            Output tensor
        """
        if self._checkpoint:
            return checkpoints[self._layer]

        return outputs[self._layer]


class Unpack(BaseLayer):
    """
    Enables a list of Tensors as input into the network, then selects which Tensor in the list to
    output.

    Methods
    -------
    forward(_, outputs, **_) -> Tensor
        Forward pass of Unpack
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            index: int,
            shapes: list[list[int] | list[list[int]]],
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        index : int
            Index of input Tensor list
        shapes : list[list[int] | list[list[int]]]
            Shape of the outputs from each layer

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._idx: int = index

        if isinstance(shapes[0][0], int):
            raise ValueError(f'Network input shape must be a list of input shapes, input shape is '
                             f'{shapes[0]}')

        shapes.append(shapes[0][self._idx])

    def forward(self, *_: Any, outputs: list[list[Tensor] | Tensor], **__: Any) -> Tensor:
        """
        Forward pass of the skip layer

        Parameters
        ----------
        outputs : list[list[(N,...) Tensor] | (N,...) Tensor]
            Output from each layer

        Returns
        -------
        (N,...) Tensor
            Output tensor
        """
        return outputs[0][self._idx]

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'index={self._idx}'
