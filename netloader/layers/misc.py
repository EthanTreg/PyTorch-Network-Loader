"""
Miscellaneous network layers
"""
import torch
import numpy as np
from torch import Tensor

from netloader.layers.utils import BaseLayer, BaseMultiLayer


class Checkpoint(BaseLayer):
    """
    Constructs a layer to create a checkpoint for using the output from the previous layer in
    future layers

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, net_check) -> Tensor
        Forward pass of the checkpoint layer
    """
    def __init__(self, shapes: list[list[int]], check_shapes: list[list[int]], **kwargs):
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
        self._check_num: int
        self._check_num = len(check_shapes)
        shapes.append(shapes[-1].copy())
        check_shapes.append(shapes[-1].copy())

    def forward(self, x: Tensor, checkpoints: list[Tensor], **_) -> Tensor:
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
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, outputs, net_check) -> Tensor
        Forward pass of the concatenation layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            net_check: bool,
            idx: int,
            layer: int,
            shapes: list[list[int]],
            check_shapes: list[list[int]],
            checkpoint: bool = False,
            dim: int = 0,
            **kwargs):
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        idx : int
            Layer number
        layer : int
            Layer index to concatenate the previous layer output with
        shapes : list[list[int]]
            Shape of the outputs from each layer
        check_shapes : list[list[int]]
            Shape of the outputs from each checkpoint
        checkpoint : bool, default = False
            If layer index should be relative to checkpoint layers
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
            idx=idx,
            **kwargs,
        )
        self._dim: int
        shape: list[int]

        self._dim = dim
        shape = shapes[-1].copy()

        # If tensors cannot be concatenated along the specified dimension
        if ((self._target[:self._dim] + self._target[self._dim + 1:] !=
             shape[:self._dim] + shape[self._dim + 1:]) or
                (len(self._target) != len(shape))):
            raise ValueError(f'Shape mismatch, input shape {shape} in layer {idx} does not match '
                             f'the target shape {self._target} in layer/checkpoint '
                             f'{self._layer} for concatenation over dimension {self._dim}')

        shape[self._dim] = shape[self._dim] + self._target[self._dim]
        shapes.append(shape)

    def forward(self, x: Tensor, outputs: list[Tensor], checkpoints: list[Tensor], **_) -> Tensor:
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


class Index(BaseLayer):
    """
    Constructs a layer to slice the last dimension from the output from the previous layer

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the indexing layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            number: int,
            shapes: list[list[int]] | None = None,
            greater: bool = True,
            **kwargs):
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
        super().__init__(**({'idx': 0} | kwargs))
        self._greater: bool
        self._number: int
        self._greater = greater
        self._number = number

        # If not used as a layer in a network
        if not shapes:
            return

        shapes.append(shapes[-1].copy())

        # Length of slice
        if (self._greater and self._number < 0) or (not self._greater and self._number > 0):
            shapes[-1][-1] = abs(self._number)
        else:
            shapes[-1][-1] = shapes[-1][-1] - abs(number)

    def forward(self, x: Tensor, **_) -> Tensor:
        """
        Forward pass of the indexing layer

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
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


class Reshape(BaseLayer):
    """
    Constructs a reshaping layer to change the data dimensions

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x)
        Forward pass of Reshape
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(self, shape: list[int], shapes: list[list[int]] | None = None, **kwargs):
        """
        Parameters
        ----------
        shape : list[int]
            Desired shape of the output tensor, ignoring first dimension
        shapes : list[list[int]], optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**({'idx': 0} | kwargs))
        self._shape: list[int]
        fixed_shape: np.ndarray

        self._shape = shape

        # If not used as a layer in a network
        if not shapes:
            return

        # If -1 in output shape, calculate the dimension length from the input dimensions
        if -1 not in self._shape:
            shapes.append(self._shape)
        elif shape.count(-1) == 1:
            shape = self._shape.copy()
            fixed_shape = np.delete(shape, np.array(shape) == -1)
            shape[shape.index(-1)] = np.prod(shapes[-1]) // np.prod(fixed_shape)
            shapes.append(shape)
        else:
            raise ValueError(f'Cannot infer output shape as -1 occurs more than once in '
                             f'{self._shape}')

        # If input tensor cannot be reshaped into output shape
        if np.prod(shapes[-1]) != np.prod(shapes[-2]):
            raise ValueError(f'Output size does not match input size for input shape {shapes[-2]} '
                             f'and output shape {shapes[-1]}')

    def forward(self, x: Tensor, **_) -> Tensor:
        """
        Forward pass of reshaping tensors

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
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


class Shortcut(BaseMultiLayer):
    """
    Constructs a shortcut layer to add the outputs from two layers

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, outputs, net_check) -> Tensor
        Forward pass of the shortcut layer
    """
    def __init__(
            self,
            net_check: bool,
            idx: int,
            layer: int,
            shapes: list[list[int]],
            check_shapes: list[list[int]],
            checkpoint: bool = False,
            **kwargs):
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        idx : int
            Layer number
        layer : int
            Layer index to concatenate the previous layer output with
        shapes : list[list[int]]
            Shape of the outputs from each layer
        check_shapes : list[list[int]]
            Shape of the outputs from each checkpoint
        checkpoint : bool, default = False
            If layer index should be relative to checkpoint layers

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            net_check,
            layer,
            shapes,
            check_shapes,
            checkpoint=checkpoint,
            idx=idx,
            **kwargs,
        )
        shape: np.ndarray
        mask: np.ndarray
        idxs: np.ndarray

        shape = np.array(shapes[-1].copy())
        mask = (shape != 1) & (self._target != 1)

        if not np.array_equal(shape[mask], np.array(self._target)[mask]):
            raise ValueError(f'Tensor shapes {shape} in layer {idx} and {self._target} in layer '
                             f'{layer} not compatible for addition.')

        # If input has any dimensions of length one, output will take the target dimension
        if 1 in shape:
            idxs = np.where(shape == 1)[0]
            shape[idxs] = self._target[idxs]

        # If target has any dimensions of length one, output will take the input dimension
        if 1 in self._target:
            idxs = np.where(self._target == 1)[0]
            shape[idxs] = shape[idxs]

        shapes.append(shape.tolist())

    def forward(self, x: Tensor, outputs: list[Tensor], checkpoints: list[Tensor], **_) -> Tensor:
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
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, outputs, net_check) -> Tensor
        Forward pass of the shortcut layer
    """
    def __init__(
            self,
            net_check: bool,
            layer: int,
            shapes: list[list[int]],
            check_shapes: list[list[int]],
            checkpoint: bool = False,
            **kwargs):
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
            If layer index should be relative to checkpoint layers

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

    def forward(self, x: Tensor, outputs: list[Tensor], checkpoints: list[Tensor], **_) -> Tensor:
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
    forward(x)
        Forward pass of Unpack
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(self, idx: int, index: int, shapes: list[list[int | list[int]]], **kwargs):
        """
        Parameters
        ----------
        idx : int
            Layer number
        index : int
            Index of input Tensor list
        shapes : list[list[int | list[int]]
            Shape of the outputs from each layer

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        self._idx: int
        self._idx = index

        if isinstance(shapes[0][0], int):
            raise ValueError(f'Network input shape must be a list of input shapes for Unpack '
                             f'layer, input shape is {shapes[0]}.')

        shapes.append(shapes[0][self._idx])

    def forward(self, _, outputs: list[list[Tensor] | Tensor], **__) -> Tensor:
        """
        Forward pass of the skip layer

        Parameters
        ----------
        outputs : list[list[Tensor] | Tensor]
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
