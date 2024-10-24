"""
Normalizing flow network layers
"""
from typing import Any

from zuko.flows import NSF
from zuko.distributions import NormalizingFlow
from torch import Tensor

from netloader.layers.base import BaseLayer


class SplineFlow(BaseLayer):
    """
    Neural spline flow layer constructor

    Should only be used as the last layer in the network as it does not return a Tensor

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers to loop through in the forward pass

    Methods
    -------
    forward(x, **_) -> NormalizingFlow
        Forward pass of the neural spline flow to return a distribution
    """
    def __init__(
            self,
            transforms: int,
            hidden_features: list[int],
            net_out: list[int],
            shapes: list[list[int]],
            context: bool = False,
            features: int | None = None,
            factor: float | None = None,
            **kwargs: Any):
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
        shapes : list[list[int]]
            Shape of the outputs from each layer
        context : bool, default = False
            If the output from the previous layer should be used to condition the normalizing flow
        features : int, optional
            Dimensions of the probability distribution, if factor is provided, features will not be
            used
        factor : float, optional
            Output features is equal to the factor of the network's output, will be used if
            provided, else features will be used
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
            features=features,
            context=context_,
            transforms=transforms,
            hidden_features=hidden_features,
        )

        shapes.append(shape)

    def forward(self, x: Tensor, **_: Any) -> NormalizingFlow:
        """
        Forward pass of the neural spline flow layer

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        NormalizingFlow
            Normalising flow distribution
        """
        if self._context:
            return self._layer(x)

        return self._layer()
