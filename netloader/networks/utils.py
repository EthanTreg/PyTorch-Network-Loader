"""
Utility mixin for network classes and helper functions for optimisers and schedulers.
"""
import pickle
from typing import Any

from torch import optim
from torch.optim.optimizer import ParamsT

from netloader.utils.types import TensorT


class UtilityMixin:
    """
    Mixin class providing utility functions for network classes.

    Methods
    -------
    set_optimiser(parameters, **kwargs)
    set_scheduler(optimiser, **kwargs)
    """

    @staticmethod
    def _data_loader_translation(low_dim: TensorT, high_dim: TensorT) -> tuple[TensorT, TensorT]:
        """
        Orders low and high dimensional tensors from the data loader as inputs and targets for the
        network.

        Parameters
        ----------
        low_dim : TensorT
            Low dimensional data from the data loader of shape (N,...) and type float, where N is
            the number of elements
        high_dim : TensorT
            High dimensional data from the data loader of shape (N,...) and type float

        Returns
        -------
        tuple[TensorT, TensorT]
            Input and output target tensors of shape (N,...) and type float
        """
        return high_dim, low_dim

    @staticmethod
    def _save_predictions(path: str, data: dict[str, Any]) -> None:
        """
        Saves network predictions to pickle file if path is provided.

        Parameters
        ----------
        path : str
            Path to save network predictions
        data : dict[str, Any]
            Network predictions of shape (N,...) to save
        """
        if not path:
            return

        with open(path + '' if '.pkl' in path else '.pkl', 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def set_optimiser(parameters: ParamsT, **kwargs: Any) -> optim.Optimizer:
        """
        Sets the optimiser for the network, by default AdamW.

        Parameters
        ----------
        parameters : ParamsT
            Network parameters to optimise
        **kwargs
            Optional keyword arguments to pass to the optimiser

        Returns
        -------
        optim.Optimizer
            Network optimiser
        """
        return optim.AdamW(parameters, **kwargs)

    @staticmethod
    def set_scheduler(optimiser: optim.Optimizer, **kwargs: Any) -> optim.lr_scheduler.LRScheduler:
        """
        Sets the scheduler for the network, by default ReduceLROnPlateau.

        Parameters
        ----------
        optimiser : optim.Optimizer
            Network optimiser
        **kwargs
            Optional keyword arguments to pass to the scheduler

        Returns
        -------
        optim.lr_scheduler.LRScheduler
            Optimiser scheduler
        """
        return optim.lr_scheduler.ReduceLROnPlateau(optimiser, **kwargs)
