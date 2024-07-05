"""
Linear network layers
"""
from __future__ import annotations
import logging as log
from typing import TYPE_CHECKING, Any, Self

import torch
import numpy as np
from torch import nn, Tensor

from netloader.layers.misc import Reshape
from netloader.layers.utils import BaseLayer

if TYPE_CHECKING:
    from netloader.network import Network


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
            factor: int | None = None,
            batch_norm: bool = False,
            activation: bool = True,
            dropout: float = 0.01,
            **kwargs: Any):
        """
        Parameters
        ----------
        net_out : list[int], optional
            Shape of the network's output, required only if layer contains factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        features : int, optional
            Number of output features for the layer,
            if out_shape from kwargs and factor is provided, features will not be used
        factor : float, optional
            Output features is equal to the factor of the network's output,
            will be used if provided, else features will be used
        batch_norm : bool, default = False
            If batch normalisation should be used
        activation : bool, default = True
            If SELU activation should be used
        dropout : float, default =  0.01
            Probability of dropout
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._activation: bool = activation
        self._batch_norm: bool = batch_norm
        self._dropout: float = dropout
        self._in_shape: list[int] = shapes[-1].copy()
        self._out_shape: list[int]

        # Remove channels dimension
        if len(self._in_shape) > 1:
            self.layers.append(Reshape([-1]))

        # Number of features can be defined by either a factor of the output size or explicitly
        if factor:
            features = max(1, int(np.prod(net_out) * factor))

        assert isinstance(features, int)
        self.layers.append(nn.Linear(
            in_features=int(np.prod(self._in_shape)),
            out_features=features,
        ))

        # Optional layers
        if self._activation:
            self.layers.append(nn.SELU())

        if self._batch_norm:
            self.layers.append(nn.BatchNorm1d(features))

        if self._dropout:
            self.layers.append(nn.Dropout(self._dropout))

        # Add channels dimension equal to input channels if input contains channels
        if len(self._in_shape) == 1:
            self._out_shape = [features]
        elif features % self._in_shape[0] == 0:
            self._out_shape = [self._in_shape[0], features // self._in_shape[0]]
            self.layers.append(Reshape(self._out_shape))
        else:
            self._out_shape = [1, features]
            self.layers.append(Reshape(self._out_shape))

        shapes.append(self._out_shape)


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
        if not self.training:
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
        self.sample_layer: torch.distributions.Distribution = torch.distributions.Normal(
            torch.tensor(0.).to(self._device),
            torch.tensor(1.).to(self._device),
        )

        if shapes[-1][0] % 2 == 1:
            log.warning(f'Sample layer in layer {idx} expects an even length along the first '
                        f'dimension, but input shape is {shapes[-1]}, last element will be ignored')

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

        net.kl_loss += net.kl_loss_weight * 0.5 * torch.mean(
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
            scale: float | tuple[float, ...] = 2,
            shape: list[int] | None = None,
            mode: str = 'nearest',
            **kwargs: Any):
        """
        Parameters
        ----------
        idx : int
            Layer number
        shapes : list[list[int]]
            Shape of the outputs from each layer
        scale : float | tuple[float], default = 2
            Factor to upscale all or individual dimensions, first dimension is ignored, won't be
            used if shape is provided
        shape : list[int], optional
            Shape of the output, will be used if provided, else factor will be used
        mode : {'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'}
            What interpolation method to use for upsampling
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(idx=idx, **kwargs)
        self._mode: str = mode
        self._scale: dict[str, list[int] | tuple[float, ...]]
        modes: dict[str, list[int]] = {
            'nearest': [2, 3, 4],
            'linear': [2],
            'bilinear': [3],
            'bicubic': [3],
            'trilinear': [4],
        }

        if not 1 < len(shapes[-1]) < 5:
            raise ValueError(f'Upsample in layer {idx} does not support tensors with more than '
                             f'4 dimensions or less than 2, input shape is {shapes[-1]}')

        if shape is not None and len(shape) != 1 and len(shape) + 1 != len(shapes[-1]):
            raise ValueError(f'Upsampling target shape {shape} is not compatible with input shape: '
                             f'{shapes[-1]}, check that channels dimension is not in target shape')

        if self._mode not in modes:
            log.warning(
                f'Upsampling method {self._mode} in layer {idx} is not supported, nearest '
                f'method will be used'
            )
            self._mode = 'nearest'

        if len(shapes[-1]) not in modes[self._mode]:
            raise ValueError(f'Upsampling method {self._mode} in layer {idx} only supports input '
                             f'dimensions of {modes[self._mode]}, input shape is {shapes[-1]}')

        if isinstance(scale, list):
            scale = tuple(scale)
        elif not isinstance(scale, tuple):
            scale = (scale,) * len(shapes[-1][1:])

        if shape:
            self._scale = {'size': shape}
            shapes.append(shape)
        else:
            self._scale = {'scale_factor': scale}
            shapes.append(shapes[-1].copy())
            shapes[-1][1:] = [
                int(length * factor) for length, factor in zip(shapes[-1][1:], scale)
            ]

        self.layers.append(nn.Upsample(**self._scale, mode=self._mode))
