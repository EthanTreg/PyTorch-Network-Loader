.. _schedulers:

**********
Schedulers
**********

PyTorch Network Loader provides the :class:`~netloader.schedulers.BaseScheduler` class along with some common schedulers
for PyTorch safe weights and dynamic parameter updates during training, such as learning rate scheduling.

Custom Schedulers
=================

NetLoader provides two base scheduler classes:

- :class:`~netloader.schedulers.BaseFuncScheduler` - Schedulers that update parameters based on a function of the
  current step.
- :class:`~netloader.schedulers.BaseScheduler` - Schedulers with more complex behavior that cannot be expressed by a
  function.

The :class:`~netloader.schedulers.BaseScheduler` class has the following methods:

.. automethod:: netloader.schedulers.BaseScheduler.step
    :noindex:

.. automethod:: netloader.schedulers.BaseScheduler.get_step
    :noindex:

Along with the magic method:

.. automethod:: netloader.schedulers.BaseScheduler.__call__
    :noindex:

Schedulers are updated via the ``step`` method with the current epoch argument.
The current value can be obtained from calling the scheduler object.
The method ``get_step`` combines the functionality of both of these methods by updating the scheduler and returning the
current value for the given epoch.

All custom schedulers must implement the ``step`` method, along with the ``__call__``, ``__getstate__``, &
``__setstate__`` methods if the custom scheduler inherits from :class:`~netloader.schedulers.BaseScheduler` or extra
functionality is required.

Custom Schedulers from :class:`~netloader.schedulers.BaseFuncScheduler`
-----------------------------------------------------------------------

The :class:`~netloader.schedulers.BaseFuncScheduler` class provides the base functionality for all schedulers that
update parameters based on a function of the current step.
When creating a custom scheduler from :class:`~netloader.schedulers.BaseFuncScheduler`, the only requirement is to
define the ``step`` method, which should define the function.
If extra attributes are required, then the ``__getstate__`` and ``__setstate__`` methods should also be defined with the
additional attributes.

The :class:`~netloader.schedulers.BaseFuncScheduler` class is initialised with the following parameters:

.. automethod:: netloader.network.Network.__init__
    :noindex:

The attributes of the :class:`~netloader.schedulers.BaseFuncScheduler` object include:

.. autoclass:: netloader.schedulers.BaseFuncScheduler
    :no-members:
    :class-doc-from: class
    :no-index:

Example implementing a linear scheduler from :class:`~netloader.schedulers.BaseFuncScheduler`:

.. code-block:: python

    from netloader.schedulers import BaseFuncScheduler

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
        def step(self, epoch: float) -> None:
            progress: float

            if epoch >= self.max_epochs:
                self._value = self.final_val
            elif epoch <= self.warmup:
                self._value = self.initial_val
            else:
                progress = (epoch - self.warmup) / (self.max_epochs - self.warmup)
                self._value = progress * (self.final_val - self.initial_val) + self.initial_val

Custom Schedulers from :class:`~netloader.schedulers.BaseScheduler`
-------------------------------------------------------------------

If your custom scheduler requires more complex functionality, then you can inherit from
:class:`~netloader.schedulers.BaseScheduler` and implement the ``step``, ``__call__``, ``__getstate__``, &
``__setstate__`` methods.

Example implementing a scheduler from :class:`~netloader.schedulers.BaseScheduler` that updates the parameter every
:math:`N` epochs.
The scheduler multiplies the initial value by a factor of :math:`\\gamma` every :math:`N` epochs for a total of
:math:`M` updates.

.. code-block:: python

    class StepScheduler(BaseScheduler):
        """
        Step scheduler.

        Attributes
        ----------
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
                max_steps: int,
                step_size: int,
                gamma: float,
                initial_value: float = 1) -> None:
            """
            Parameters
            ----------
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
            self.max_steps: int = max_steps
            self.step_size: int = step_size
            self.gamma: float = gamma
            self.initial_val: float = initial_value
            self._value: float = self.initial_val

        def __call__(self) -> float:
            return self._value

        def __getstate__(self) -> dict[str, Any]:
            # Get the scheduler parameters for saving
            return {
                'max_steps': self.max_steps,
                'step_size': self.step_size,
                'gamma': self.gamma,
                'initial_val': self.initial_val,
            }

        def __setstate__(self, state: dict[str, Any]) -> None:
            # Set the scheduler parameters for loading
            self.__dict__.update(state)
            self._value = self.initial_val

        def step(self, epoch: int | float) -> None:
            # Number of steps given the current epoch
            steps: int = min(max(int((epoch - self.warmup) // self.step_size), 0), self.max_steps)

            if not steps:
                self._value = self.initial_val
                return

            self._value = self.initial_val * (self.gamma ** steps)

Built-In Schedulers
===================

All schedulers currently supported by NetLoader are listed below with a brief description of their functionality and
their initialisation arguments, excluding general initialisation arguments defined in
:class:`~netloader.schedulers.BaseScheduler`.
For more details on each loss function, see :ref:`API documentation <api_schedulers>`.

.. autosig:: netloader.schedulers
