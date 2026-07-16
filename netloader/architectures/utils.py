"""
Utility mixin for architecture classes and helper functions for optimisers and schedulers.
"""
import pickle
import logging as log
from warnings import warn
from typing import Any, Callable

from torch import optim, Tensor
from torch.optim.optimizer import ParamsT

from netloader.utils.types import Param, TensorT

logger: log.Logger = log.getLogger(__name__)


class UtilityMixin:
    """
    Mixin class providing utility functions for architecture classes.
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
    def _param_device(params: dict[Tensor, Param] | Param, *args: Any, **kwargs: Any) -> None:
        """
        Sends parameters to the device, such as the parameters in the optimiser.

        Parameters
        ----------
        params : dict[Tensor, Param] | Param
            Parameters to move/cast
        *args, **kwargs
            Arguments to pass to torch.Tensor.to
        """
        param: Tensor | Param

        for param in params.values():
            if isinstance(param, dict):
                UtilityMixin._param_device(param, *args, **kwargs)
            elif isinstance(param, Tensor):
                param.data = param.data.to(*args, **kwargs)

                # pylint: disable=protected-access
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(*args, **kwargs)
                # pylint: enable=protected-access

    @staticmethod
    def _save_predictions(path: str, data: dict[str, Any]) -> None:
        """
        Saves architecture predictions to pickle file if path is provided.

        Parameters
        ----------
        path : str
            Path to save architecture predictions
        data : dict[str, Any]
            Architecture predictions of shape (N,...) to save
        """
        if not path:
            return

        with open(path + '' if '.pkl' in path else '.pkl', 'wb') as file:
            pickle.dump(data, file)

    def _method_override(self, name: str, owner: type) -> bool:
        """
        Checks if a method is overridden in a subclass.

        Parameters
        ----------
        name : str
            Name of the method to check
        owner : type
            Class to check for method override

        Returns
        -------
        bool
            True if the method is overridden, False otherwise
        """
        method: Callable | None = getattr(self, name, None)

        if method is None:
            logger.warning(f'Method {name} not found in {type(self).__name__}')
            return False
        return method.__qualname__.split('.')[0] != owner.__name__

    @staticmethod
    def init_optimiser(parameters: ParamsT, **kwargs: Any) -> optim.Optimizer:
        """
        Sets the optimiser for the architecture, by default AdamW.

        Parameters
        ----------
        parameters : ParamsT
            Architecture parameters to optimise
        **kwargs
            Optional keyword arguments to pass to the optimiser

        Returns
        -------
        Optimizer
            Architecture optimiser
        """
        return optim.AdamW(parameters, **kwargs)

    @staticmethod
    def init_scheduler(optimiser: optim.Optimizer, **kwargs: Any) -> optim.lr_scheduler.LRScheduler:
        """
        Initialise the scheduler for the architecture, by default ReduceLROnPlateau.

        Parameters
        ----------
        optimiser : Optimizer
            Architecture optimiser
        **kwargs
            Optional keyword arguments to pass to the scheduler

        Returns
        -------
        LRScheduler
            Optimiser scheduler
        """
        return optim.lr_scheduler.ReduceLROnPlateau(optimiser, **kwargs)

    @staticmethod
    def set_optimiser(parameters: ParamsT, **kwargs: Any) -> optim.Optimizer:
        """
        Deprecated alias of :meth:`init_optimiser`.

        .. deprecated:: 3.11.0
           Use :meth:`init_optimiser` instead.
        """
        warn(
            "UtilityMixin.set_optimiser is deprecated, use UtilityMixin.init_optimiser "
            "instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return UtilityMixin.init_optimiser(parameters, **kwargs)

    @staticmethod
    def set_scheduler(optimiser: optim.Optimizer, **kwargs: Any) -> optim.lr_scheduler.LRScheduler:
        """
        Deprecated alias of :meth:`init_scheduler`.

        .. deprecated:: 3.11.0
           Use :meth:`init_scheduler` instead.
        """
        warn(
            "UtilityMixin.set_scheduler is deprecated, use UtilityMixin.init_scheduler "
            "instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return UtilityMixin.init_scheduler(optimiser, **kwargs)


__all__ = ['UtilityMixin']
