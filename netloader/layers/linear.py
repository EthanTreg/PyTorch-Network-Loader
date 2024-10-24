"""
Linear network layers
"""
from __future__ import annotations
import logging as log
from typing import TYPE_CHECKING, Any, Self

import torch
import numpy as np
from torch import nn, Tensor

from netloader.layers.base import BaseLayer

if TYPE_CHECKING:
    from netloader.network import Network


class Activation(BaseLayer):
    """
    Activation layer constructor

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass
    """
    def __init__(
            self,
            activation: str = 'ELU',
            shapes: list[list[int]] | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        activation : str, default = 'ELU'
            Which activation function to use from PyTorch
        shapes : list[list[int]], optional
            Shape of the outputs from each layer, only required if tracking layer outputs is
            necessary

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**({'idx': 0} | kwargs))
        self.layers.add_module('activation', getattr(nn, activation)())

        # If not used as a layer in a network
        if not shapes:
            return

        shapes.append(shapes[-1].copy())


class Linear(BaseLayer):
    """
    Linear layer constructor

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the linear layer
    """
    def __init__(
            self,
            net_out: list[int],
            shapes: list[list[int]],
            features: int | None = None,
            layer: int | None = None,
            factor: float | None = None,
            batch_norm: bool = False,
            dropout: float = 0,
            activation: str | None = 'SELU',
            **kwargs: Any):
        """
        Parameters
        ----------
        net_out : list[int]
            Shape of the network's output
        shapes : list[list[int]]
            Shape of the outputs from each layer
        features : int, optional
            Number of output features for the layer, if factor is provided, features will not be
            used
        layer : int, optional
            If factor is not None, which layer for factor to be relative to, if None, network output
            will be used
        factor : float, optional
            Output features is equal to the factor of the network's output, or if layer is provided,
            which layer to be relative to, will be used if provided, else features will be used
        batch_norm : bool, default = False
            If batch normalisation should be used
        dropout : float, default =  0
            Probability of dropout
        activation : str | None, default = 'SELU'
            Which activation function to use from PyTorch
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        shape: list[int] = shapes[-1].copy()
        target: list[int] = shapes[layer] if layer is not None else net_out

        # Number of features can be defined by either a factor of the output size or explicitly
        shape = self._check_factor_filters(shape[::-1], features, factor, target[::-1])[::-1]

        self.layers.add_module('linear', nn.Linear(
            in_features=shapes[-1][-1],
            out_features=shape[-1],
        ))

        # Optional layers
        if activation:
            self.layers.add_module('activation', getattr(nn, activation)())

        if batch_norm:
            self.layers.add_module('batch_norm', nn.BatchNorm1d(shape[0]))

        if dropout:
            self.layers.add_module('dropout', nn.Dropout(dropout))

        shapes.append(shape)


class OrderedBottleneck(BaseLayer):
    """
    Information-ordered bottleneck to randomly change the size of the bottleneck in an autoencoder
    to encode the most important information in the first values of the latent space.

    See `Information-Ordered Bottlenecks for Adaptive Semantic Compression
    <https://arxiv.org/abs/2305.11213>`_ by Ho et al. (2023)

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass
    min_size : int, default = 0
        Minimum gate size

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the information-ordered bottleneck layer
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(self, shapes: list[list[int]], min_size: int = 0, **kwargs: Any):
        """
        Parameters
        ----------
        shapes : list[list[int]]
            Shape of the outputs from each layer
        min_size : int, default = 0
            Minimum gate size
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self.min_size: int = min_size
        shapes.append(shapes[-1].copy())

    def forward(self, x: Tensor, **_: Any) -> Tensor:
        """
        Forward pass of the information-ordered bottleneck layer

        Parameters
        ----------
        x : (N,...,Z) Tensor
            Input tensor with batch size N and latent Z where Z > min_size

        Returns
        -------
        (N,...,Z) Tensor
            Input zeroed from a random index to the last value along the dimension Z
        """
        if not self.training or self.min_size >= x.size(-1):
            return x

        idx: int = np.random.randint(self.min_size, x.size(-1))
        gate: Tensor = torch.zeros(x.size(-1)).to(self._device)
        gate[:idx + 1] = 1
        return x * gate

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return f'min_size={self.min_size}'


class Sample(BaseLayer):
    """
    Samples random values from a Gaussian distribution for a variational autoencoder,
    mean and standard deviation are the first and second half of the input channels with the last
    channel ignored if there are an odd number

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass
    sample_layer : Normal
        Layer to sample values from a Gaussian distribution

    Methods
    -------
    forward(x, net) -> Tensor
        Forward pass of the sampling layer
    """
    def __init__(self, idx: int, shapes: list[list[int]], **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        self.sample_layer: torch.distributions.Normal = torch.distributions.Normal(
            torch.tensor(0.).to(self._device),
            torch.tensor(1.).to(self._device),
        )

        if shapes[-1][0] % 2 == 1:
            log.warning(f'Sample in layer {idx} expects an even length along the first dimension, '
                        f'but input shape is {shapes[-1]}, last element will be ignored')

        shapes.append(shapes[-1].copy())
        shapes[-1][0] = shapes[-1][0] // 2

    def forward(self, x: Tensor, net: Network, **_: Any) -> Tensor:
        """
        Forward pass of the sampling layer for a variational autoencoder

        Parameters
        ----------
        x : (N,C,...) | (N,Z) Tensor
            Input tensor with batch size N containing the mean and standard deviation in the
            channels dimension C, or latent Z
        net : Network
            Parent network that this layer is part of

        Returns
        -------
        (N,C/2,...) | (N,Z/2) Tensor
            Output tensor sampled from the input tensor split into mean and standard deviation
        """
        split: int = x.size(1) // 2
        mean: Tensor = x[:, :split]
        std: Tensor = torch.exp(x[:, split:2 * split])
        x = mean + std * self.sample_layer.sample(mean.shape)

        net.kl_loss = 0.5 * torch.mean(
            mean ** 2 + std ** 2 - 2 * torch.log(std) - 1
        )
        return x

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self.sample_layer.loc = self.sample_layer.loc.to(*args, **kwargs)
        self.sample_layer.scale = self.sample_layer.scale.to(*args, **kwargs)
        return self


class Upsample(BaseLayer):
    """
    Constructs an upsampler

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the upsampling layer
    """
    def __init__(
            self,
            idx: int,
            shapes: list[list[int]],
            shape: list[int] | None = None,
            scale: float | tuple[float, ...] = 2,
            mode: str = 'nearest',
            **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        shape : list[int], optional
            Shape of the output, will be used if provided, else scale will be used
        scale : float | tuple[float], default = 2
            Factor to upscale all or individual dimensions, first dimension is ignored, won't be
            used if shape is provided
        mode : {'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'}
            What interpolation method to use for upsampling
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        scale_arg: dict[str, tuple[int, ...] | tuple[float, ...]]
        modes: dict[str, list[int]] = {
            'nearest': [2, 3, 4],
            'linear': [2],
            'bilinear': [3],
            'bicubic': [3],
            'trilinear': [4],
        }

        # Check for errors
        self._check_shape(shapes[-1])
        self._check_upsample(shapes[-1], shape)
        self._check_options('mode', mode, set(modes))
        self._check_mode_dimension(mode, shapes[-1], modes)

        if isinstance(scale, list):
            scale = tuple(scale)
        elif not isinstance(scale, tuple):
            scale = (scale,) * len(shapes[-1][1:])

        if shape:
            scale_arg = {'size': tuple(shape)}
            shapes.append(shape)
        else:
            scale_arg = {'scale_factor': scale}
            shapes.append(shapes[-1].copy())
            shapes[-1][1:] = [
                int(length * factor) for length, factor in zip(shapes[-1][1:], scale)
            ]

        self.layers.add_module('upsample', nn.Upsample(**scale_arg, mode=mode))

    @staticmethod
    def _check_mode_dimension(mode: str, shape: list[int], modes: dict[str, list[int]]) -> None:
        """
        Checks if the upsampling mode supports the number of input dimensions

        Parameters
        ----------
        mode : str
            Current upsampling mode
        shape : list[int]
            Input shape
        modes : dict[str, list[int]]
            Modes with a list of supported dimensions
        """
        if len(shape) not in modes[mode]:
            raise ValueError(f'{mode} only supports input dimensions of {modes[mode]}, '
                             f'input shape is {shape}')

    @staticmethod
    def _check_upsample(in_shape: list[int], out_shape: list[int] | None) -> None:
        """
        Checks if the input shape has the same number of dimensions as the output shape

        Parameters
        ----------
        in_shape : list[int]
            Input shape
        out_shape : list[int]
            Target output shape
        """
        if out_shape is not None and len(out_shape) != 1 and len(out_shape) + 1 != len(in_shape):
            raise ValueError(f'Target shape {out_shape} is not compatible with input shape '
                             f'{in_shape}, check that channels dimension is not in target shape')
