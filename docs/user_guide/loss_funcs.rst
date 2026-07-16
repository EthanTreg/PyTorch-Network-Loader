.. _loss_funcs:

**************
Loss Functions
**************

PyTorch Network Loader provides the :class:`~netloader.loss_funcs.BaseLoss` class along with some common loss functions
for safe weight loading in PyTorch.
These are useful if an architecture allows for dynamic loss functions (i.e. loss functions can be set at runtime rather
then being baked into the architecture).

Converting PyTorch Loss Functions to :class:`~netloader.loss_funcs.BaseLoss`
============================================================================

Many loss functions are already implemented in PyTorch; however, if an architecture saves the loss function in its
state, then the architecture cannot be loaded without disabling safe weight loading.
NetLoader provides the :class:`~netloader.loss_funcs.GeneralLoss` class to convert loss functions from :mod:`~torch.nn`
to :class:`~netloader.loss_funcs.BaseLoss` for safe weight loading within NetLoader projects.

To convert a standard PyTorch loss function to a :class:`~netloader.loss_funcs.BaseLoss` create a new class that
inherits from :class:`~netloader.loss_funcs.GeneralLoss` and in the ``__init__`` method, pass the name of the PyTorch
loss function from :mod:`~torch.nn` to ``super().__init__()`` along with any optional arguments or keyword arguments.

If the forward method of the PyTorch loss function differs from accepting the network outputs and target values as
arguments, then the ``forward`` method can be overridden.

Creating Gaussian Negative Log Likelihood Loss
----------------------------------------------

The Gaussian negative log likelihood loss (GaussianNLLLoss) is implemented in :mod:`~torch.nn` and can be converted
using :class:`~netloader.loss_funcs.GeneralLoss`.
GaussianNLL requires the targets to have uncertainties; therefore, one dimension of the targets will have size 2 to
contain the values and uncertainties and the outputs will share the dimension but with a size of 1.
Therefore, during initialisation, we will have a ``dim`` argument to specify the dimension along which the values and
uncertainties are stored.

.. code-block:: python

    from torch import nn
    from netloader.loss_funcs import GeneralLoss

    class GaussianNLLLoss(GeneralLoss):
        """
        Gaussian Negative Log Likelihood (GaussianNLL) loss function.
        """
        def __init__(self, *args: Any, dim: int = 1, **kwargs: Any) -> None:
            """
            Parameters
            ----------
            dim : int, Optional
                Dimension of the target that contains the mean and uncertainty, default = 1
            *args
                Optional arguments to be passed to GaussianNLLLoss
            **kwargs
                Optional keyword arguments to be passed to GaussianNLLLoss
            """
            super().__init__('GaussianNLLLoss', *args, **kwargs)
            self._dim: int = dim

Then we need to save and load the ``dim`` argument in the state:

.. code-block:: python

    class GaussianNLLLoss(GeneralLoss):
        ...

        def __getstate__(self) -> dict[str, Any]:
            return super().__getstate__() | {'dim': self._dim}

        def __setstate__(self, state: dict[str, Any]) -> None:
            super().__setstate__(state)
            self._dim = state['dim']

Finally, we need to override the forward method to split the targets into values and uncertainties:

.. code-block:: python

    from torch import Tensor

    class GaussianNLLLoss(GeneralLoss):
        ...

        def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        idx: list[list[int] | slice] = [slice(None)] * target.dim()
        idx[self._dim] = [0, 1]
        return self._loss_func(output[*idx][0], target[*idx][0], target[*idx][1] ** 2)

Creating Custom Loss Functions
==============================

Custom loss functions can be created by inheriting from the :class:`~netloader.loss_funcs.BaseLoss` class and passing
the custom loss function class to :class:`~netloader.loss_funcs.BaseLoss` in the ``__init__`` method.
If the custom loss function is more complex than passing the outputs and targets to the loss function's forward method,
then the ``forward`` method can be overwritten.

Creating Mean Squared Error Loss
--------------------------------

The mean squared error (MSE) loss function can be created by inheriting from :class:`~netloader.loss_funcs.BaseLoss` and
passing the :class:`~torch.nn.MSELoss` class to the ``__init__`` method:

.. code-block:: python

    from torch import nn
    from netloader.loss_funcs import BaseLoss

    class MSELoss(BaseLoss):
        """
        Mean Squared Error (MSE) loss function.
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

Built-In Loss Functions
=======================

All loss functions currently supported by NetLoader are listed below with a brief description of their functionality and
their initialisation arguments, excluding general initialisation arguments defined in
:class:`~netloader.loss_funcs.BaseLoss`.
For more details on each loss function, please refer to the :ref:`API documentation <api_loss_funcs>`.

.. autoloss:: netloader.loss_funcs
