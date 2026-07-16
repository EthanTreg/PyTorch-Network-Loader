"""
Scheduler classes.
"""
from typing import Any
from abc import ABC, abstractmethod

import torch
import numpy as np


class BaseScheduler(ABC):
    """
    Abstract base class for schedulers.
    """
    def __init__(self) -> None:
        # Adds all scheduler classes to list of safe PyTorch classes when loading saved
        # architectures
        torch.serialization.add_safe_globals([self.__class__])

    @abstractmethod
    def __call__(self) -> float:
        """
        Returns the current value of the scheduler.
        """

    @abstractmethod
    def __getstate__(self) -> dict[str, Any]:
        """
        Returns the state of the scheduler.

        Returns
        -------
        dict[str, Any]
            State of the scheduler
        """

    @abstractmethod
    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the scheduler.

        Parameters
        ----------
        state : dict[str, Any]
            State of the scheduler
        """

    def __repr__(self) -> str:
        """
        Returns the string representation of the scheduler.

        Returns
        -------
        str
            String representation of the scheduler
        """
        return (f'{self.__class__.__name__}(' +
                (f'{self.extra_repr()})' if self.extra_repr() else ')'))

    @abstractmethod
    def step(self, epoch: int | float) -> None:
        """
        Steps the scheduler.

        Parameters
        ----------
        epoch : int | float
            Current epoch
        """

    def extra_repr(self) -> str:
        """
        Additional string representation of the scheduler.

        Returns
        -------
        str
            Additional string representation of the scheduler
        """
        return ''

    def get_step(self, epoch: int | float) -> float:
        """
        Returns the current value of the scheduler for the given epoch.

        Parameters
        ----------
        epoch : int | float
            Current epoch

        Returns
        -------
        float
            Current value of the scheduler for the given epoch
        """
        self.step(epoch)
        return self()


class BaseFuncScheduler(BaseScheduler, ABC):
    """
    Abstract base class for simple function schedulers.

    Attributes
    ----------
    max_epochs : int
        Maximum number of epochs
    warmup : int
        Number of warmup epochs
    initial_val : float
        Initial value of the scheduler
    final_val : float
        Final value of the scheduler
    """
    max_epochs: int
    warmup: int
    final_val: float
    initial_val: float
    _value: float

    def __init__(
            self,
            max_epochs: int,
            *,
            warmup_epochs: int = 0,
            final_value: float = 1,
            initial_value: float = 0) -> None:
        """
        Parameters
        ----------
        max_epochs : int
            Maximum number of epochs
        warmup_epochs : int, Optional
            Number of warmup epochs, default = 0
        final_value : float, Optional
            Maximum value of the scheduler, default = 1
        initial_value : float, Optional
            Minimum value of the scheduler, default = 0
        """
        super().__init__()
        self.max_epochs = max_epochs
        self.warmup = warmup_epochs
        self.final_val = final_value
        self.initial_val = initial_value
        self._value = self.final_val
        self.step(0)

        if self.max_epochs < warmup_epochs:
            raise ValueError(f'max_epochs ({self.max_epochs}) should be larger than warmup_epochs '
                             f'({self.warmup})')

    def __call__(self) -> float:
        return self._value

    def __getstate__(self) -> dict[str, Any]:
        return {
            'max_epochs': self.max_epochs,
            'warmup': self.warmup,
            'final_val': self.final_val,
            'initial_val': self.initial_val,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._value = self.final_val
        self.step(0)

    def extra_repr(self) -> str:
        return (f'max_epochs={self.max_epochs}, warmup_epochs={self.warmup}, '
                f'final_value={self.final_val}, initial_value={self.initial_val}')

class LinearScheduler(BaseFuncScheduler):
    """
    Linear scheduler.

    Attributes
    ----------
    max_epochs : int
        Maximum number of epochs
    warmup : int
        Number of warmup epochs
    initial_val : float
        Initial value of the scheduler
    final_val : float
        Final value of the scheduler
    """
    def step(self, epoch: int | float) -> None:
        progress: float

        if epoch >= self.max_epochs:
            self._value = self.final_val
        elif epoch <= self.warmup:
            self._value = self.initial_val
        else:
            progress = (epoch - self.warmup) / (self.max_epochs - self.warmup)
            self._value = progress * (self.final_val - self.initial_val) + self.initial_val


class SigmoidScheduler(BaseFuncScheduler):
    """
    Sigmoid scheduler.

    Attributes
    ----------
    max_epochs : int
        Maximum number of epochs
    warmup : int
        Number of warmup epochs
    initial_val : float
        Initial value of the scheduler
    final_val : float
        Final value of the scheduler
    compression : float
        Compression factor for the sigmoid function, higher values result in a steeper curve
    """
    compression: float = 10

    def __init__(
            self,
            max_epochs: int,
            *,
            warmup_epochs: int = 0,
            final_value: float = 1,
            initial_value: float = 0,
            compression: float = 0) -> None:
        """
        Parameters
        ----------
        max_epochs : int
            Maximum number of epochs
        warmup_epochs : int, Optional
            Number of warmup epochs, default = 0
        final_value : float, Optional
            Maximum value of the scheduler, default = 1
        initial_value : float, Optional
            Minimum value of the scheduler, default = 0
        compression : float, Optional
            Compression factor for the tanh function, higher values result in a steeper curve,
            default = 10
        """
        super().__init__(
            max_epochs,
            warmup_epochs=warmup_epochs,
            final_value=final_value,
            initial_value=initial_value,
        )
        self.compression = compression or self.compression

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'compression': self.compression}

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.compression = state['compression']

    def extra_repr(self) -> str:
        return f'{super().extra_repr()}, compression={self.compression}'

    def step(self, epoch: int | float) -> None:
        progress: float

        if epoch >= self.max_epochs:
            self._value = self.final_val
        elif epoch <= self.warmup:
            self._value = self.initial_val
        else:
            progress = 1 / (1 + np.exp(
                -self.compression * ((epoch - self.warmup) / (self.max_epochs - self.warmup) - 0.5)
            ))
            self._value = progress * (self.final_val - self.initial_val) + self.initial_val


class StepScheduler(BaseScheduler):
    """
    Step scheduler.

    Attributes
    ----------
    warmup : int
        Number of warmup epochs
    max_steps : int
        Maximum number of steps
    step_size : int
        Number of epochs between each step
    gamma : float
        Multiplicative factor of the step
    initial_val : float
        Initial value of the scheduler
    """
    def __init__(
            self,
            warmup: int,
            max_steps: int,
            step_size: int,
            gamma: float,
            initial_value: float = 1) -> None:
        """
        Parameters
        ----------
        warmup : int
            Number of warmup epochs
        max_steps : int
            Maximum number of steps
        step_size : int
            Number of epochs between each step
        gamma : float
            Multiplicative factor of the step
        initial_value : float
            Initial value of the scheduler, default = 1
        """
        super().__init__()
        self.warmup: int = warmup
        self.max_steps: int = max_steps
        self.step_size: int = step_size
        self.gamma: float = gamma
        self.initial_val: float = initial_value
        self._value: float = self.initial_val

    def __call__(self) -> float:
        return self._value

    def __getstate__(self) -> dict[str, Any]:
        return {
            'warmup': self.warmup,
            'max_steps': self.max_steps,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'initial_val': self.initial_val,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._value = self.initial_val

    def extra_repr(self) -> str:
        return (f'warmup={self.warmup}, max_steps={self.max_steps}, step_size={self.step_size}, '
                f'initial_value={self.initial_val}')

    def step(self, epoch: int | float) -> None:
        steps: int = min(max(int((epoch - self.warmup) // self.step_size), 0), self.max_steps)

        if not steps:
            self._value = self.initial_val
            return

        self._value = self.initial_val * (self.gamma ** steps)


class TanhScheduler(SigmoidScheduler):
    """
    Tanh scheduler.

    Attributes
    ----------
    max_epochs : int
        Maximum number of epochs
    warmup : int
        Number of warmup epochs
    initial_val : float
        Initial value of the scheduler
    final_val : float
        Final value of the scheduler
    compression : float
        Compression factor for the tanh function, higher values result in a steeper curve
    """
    compression: float = 5

    def __init__(
            self,
            max_epochs: int,
            *,
            warmup_epochs: int = 0,
            final_value: float = 1,
            initial_value: float = 0,
            compression: float = 0) -> None:
        """
        Parameters
        ----------
        max_epochs : int
            Maximum number of epochs
        warmup_epochs : int, Optional
            Number of warmup epochs, default = 0
        final_value : float, Optional
            Maximum value of the scheduler, default = 1
        initial_value : float, Optional
            Minimum value of the scheduler, default = 0
        compression : float, Optional
            Compression factor for the sigmoid function, higher values result in a steeper curve,
            default = 5
        """
        super().__init__(
            max_epochs,
            warmup_epochs=warmup_epochs,
            final_value=final_value,
            initial_value=initial_value,
            compression=compression,
        )

    def step(self, epoch: int | float) -> None:
        progress: float

        if epoch >= self.max_epochs:
            self._value = self.final_val
        elif epoch <= self.warmup:
            self._value = self.initial_val
        else:
            progress = 2 / (1 + np.exp(
                -self.compression * (epoch - self.warmup) / (self.max_epochs - self.warmup)
            )) - 1
            self._value = progress * (self.final_val - self.initial_val) + self.initial_val


__all__ = [
    'BaseScheduler',
    'BaseFuncScheduler',
    'LinearScheduler',
    'SigmoidScheduler',
    'StepScheduler',
    'TanhScheduler',
]
