"""
Loss function classes for safe saving
"""
from typing import Callable, Any

import torch
from torch import nn, Tensor


class BaseLoss(nn.Module):
    """
    A base class for loss functions.

    This class is used to add all loss function classes to the list of safe PyTorch classes when
    loading saved networks.
    """
    def __init__(self, loss_func: type, *args: Any, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        loss_func : type
            Loss function class to be used

        *args
            Optional arguments to be passed to loss_func
        **kwargs
            Optional keyword arguments to be passed to loss_func
        """
        super().__init__()
        self._args: tuple[Any, ...] = args
        self._kwargs: dict[str, Any] = kwargs
        self._loss_func: Callable = loss_func(*self._args, **self._kwargs)

        # Adds all loss classes to list of safe PyTorch classes when loading saved networks
        torch.serialization.add_safe_globals([self.__class__])

    def __repr__(self) -> str:
        """
        Returns a string representation of the loss function

        Returns
        -------
        str
            String representation of the loss function
        """
        return self.__class__.__name__

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the loss function for pickling

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the loss function
        """
        return {'args': self._args, 'kwargs': self._kwargs}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the loss function for pickling

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the loss function
        """
        super().__init__()
        self._args = state['args']
        self._kwargs = state['kwargs']

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the loss function

        Parameters
        ----------
        output : Tensor
            Output from the network with shape (N,...), where N is the number of elements
            N predictions from the network
        target : Tensor
            Target values with shape (N,...)

        Returns
        -------
        Tensor
            Loss value with shape (1)
        """
        return self._loss_func(output, target)


class CrossEntropyLoss(BaseLoss):
    """
    Cross entropy loss function
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        *args
            Optional arguments to be passed to CrossEntropyLoss
        **kwargs
            Optional keyword arguments to be passed to CrossEntropyLoss
        """
        super().__init__(nn.CrossEntropyLoss, *args, **kwargs)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._loss_func = nn.CrossEntropyLoss(*self._args, **self._kwargs)


class GaussianNLLLoss(BaseLoss):
    """
    Gaussian negative log likelihood loss function
    """
    def __init__(self, *args: Any, dim: int = 1, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        dim : int, default = 1
            Dimension of the target that contains the mean and uncertainty
        *args
            Optional arguments to be passed to GaussianNLLLoss
        **kwargs
            Optional keyword arguments to be passed to GaussianNLLLoss
        """
        super().__init__(nn.GaussianNLLLoss, *args, **kwargs)
        self._dim: int = dim

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state['dim'] = self._dim
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._loss_func = nn.GaussianNLLLoss(*self._args, **self._kwargs)
        self._dim = state['dim']

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        if target.shape[self._dim] != 2 or output.shape[self._dim] != 1:
            raise ValueError(f'Target shape must be 2 and output shape must be 1 at dim '
                             f'{self._dim}, got target shape {target.shape} and output shape '
                             f'{output.shape}')

        idx: list[list[int] | slice] = [slice(None)] * target.dim()
        idx[self._dim] = [0, 1]
        return self._loss_func(output[*idx][0], target[*idx][0], target[*idx][1] ** 2)


class MSELoss(BaseLoss):
    """
    Mean Squared Error (MSE) loss function
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        *args
            Optional arguments to be passed to MSELoss
        **kwargs
            Optional keyword arguments to be passed to MSELoss
        """
        super().__init__(nn.MSELoss, *args, **kwargs)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._loss_func = nn.MSELoss(*self._args, **self._kwargs)


__all__ = ['CrossEntropyLoss', 'GaussianNLLLoss', 'MSELoss']
