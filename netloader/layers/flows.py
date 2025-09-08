"""
Normalizing flow network layers
"""
from typing import Any

from torch import Tensor
from zuko.flows import NSF
from zuko.distributions import NormalizingFlow

from netloader.utils import Shapes
from netloader.layers.base import BaseLayer


class SplineFlow(BaseLayer):
    """
    Neural spline flow layer constructor.

    Should only be used as the last layer in the network as it does not return a Tensor.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks

    Methods
    -------
    forward(x) -> NormalizingFlow
        Forward pass of the neural spline flow to return a distribution
    """
    def __init__(
            self,
            transforms: int,
            hidden_features: list[int],
            net_out: list[int],
            shapes: Shapes,
            *,
            context: bool = False,
            features: int | None = None,
            factor: float | None = None,
            **kwargs: Any) -> None:
        """
        Generates a neural spline flow (NSF) for use in BaseNetwork

        Adds attributes of name ('flow'), optimiser (Adam), and scheduler (ReduceLROnPlateau)

        Parameters
        ----------
        transforms : int
            Number of transforms
        hidden_features : list[int]
            Number of features in each of the hidden layers
        net_out : list[int]
            Shape of the network's output
        shapes : Shapes
            Shape of the outputs from each layer
        context : bool, default = False
            If the output from the previous layer should be used to condition the normalizing flow
        features : int | None, default = None
            Dimensions of the probability distribution, if factor is provided, features will not be
            used
        factor : float | None, default = None
            Output features is equal to the factor of the network's output, will be used if
            provided, else features will be used
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._context: bool = context
        self._layer: NSF
        context_: int
        shape: list[int] = shapes[-1].copy()

        # Number of features can be defined by either a factor of the output size or explicitly
        if factor:
            shape[-1] = max(1, int(net_out[-1] * factor))
        elif features:
            shape[-1] = features
        else:
            raise ValueError('Either features or factor must be provided and be non-zero')

        if self._context:
            context_ = shapes[-1][-1]
        else:
            context_ = 0

        self._layer = NSF(
            features=shape[-1],
            context=context_,
            transforms=transforms,
            hidden_features=hidden_features,
        )

        shapes.append(shape)

    def forward(self, x: Tensor, *_: Any, **__: Any) -> NormalizingFlow:
        """
        Forward pass of the neural spline flow layer

        Parameters
        ----------
        x : Tensor
            Input tensor with shape (N,...) and type float, where N is the batch size

        Returns
        -------
        NormalizingFlow
            Normalising flow distribution
        """
        if self._context:
            return self._layer(x)
        return self._layer()
