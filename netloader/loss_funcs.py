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
    def __init__(self, loss_func: Callable, *args: Any, **kwargs: Any):
        """
        Parameters
        ----------
        loss_func : Callable
            Loss function to be used

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
        output : (N,...) Tensor
            N predictions from the network
        target : (N,...) Tensor
            N target values

        Returns
        -------
        (1) Tensor
            Mean squared error loss
        """
        return self._loss_func(output, target)


class MSELoss(BaseLoss):
    """
    Mean Squared Error (MSE) loss function
    """
    def __init__(self, *args: Any, **kwargs: Any):
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


class CrossEntropyLoss(BaseLoss):
    """
    Cross entropy loss function
    """
    def __init__(self, *args: Any, **kwargs: Any):
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
