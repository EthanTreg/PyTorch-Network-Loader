.. _transforms:

**********
Transforms
**********

PyTorch Network Loader provides the :class:`~netloader.transforms.BaseTransform` class along with some common transforms
for PyTorch safe weights and data preprocessing.
Transforms are applied to the data before it is passed to the network for training or inference.
Transforms can be saved in architectures so normalisation is consistent between forward passes & during prediction, the
predictions get untransformed.
Transforms are Numpy/PyTorch agnostic, so either can be passed through.

Custom Transforms
=================

NetLoader provides the :class:`~netloader.transforms.BaseTransform` class for data preprocessing, inverse
transformations, and uncertainty propagation.

The :class:`~netloader.transforms.BaseTransform` class has the following methods:

.. automethod:: netloader.transforms.BaseTransform.forward
    :noindex:

.. automethod:: netloader.transforms.BaseTransform.backward
    :noindex:

.. automethod:: netloader.transforms.BaseTransform.forward_grad
    :noindex:

.. automethod:: netloader.transforms.BaseTransform.backward_grad
    :noindex:

Along with the magic method:

.. automethod:: netloader.transforms.BaseTransform.__call__
    :noindex:

All transform operations can be performed via the ``__call__`` method with the optional arguments ``back`` and
``uncertainty``.
:class:`~netloader.transforms.BaseTransform` also accepts :class:`~netloader.data.structures.Data` objects as input to transform
the data and uncertainties.
To transform data, first initialise the transform object, then call the object with the data (and optional
uncertainties) to be transformed.

.. code-block:: python

    import torch
    from torch import Tensor
    from netloader.data import Data
    from netloader import transforms

    values: Tensor
    values_2: Tensor
    trans_values: Tensor
    uncertainties: Tensor
    uncertainties_2: Tensor
    trans_uncertainties: Tensor
    data: Data
    data_2: Data
    trans_data: Data
    transform: transforms.BaseTransform

     values = torch.tensor([1.0, 2.0, 3.0])
     uncertainties = torch.tensor([0.1, 0.2, 0.3])
     data = Data(values, uncertainty=uncertainties)
     transform = transforms.Log()

    # Transform data
    trans_values, trans_uncertainties = transform(values, uncertainty=uncertainties)
    trans_data = transform(data)

    # Untransform data
    values_2, uncertainties_2 = transform(trans_values, back=True, uncertainty=trans_uncertainties)
    data_2 = transform(trans_data, back=True)

    assert torch.isclose(values, values_2).all()
    assert torch.isclose(uncertainties, uncertainties_2).all()
    assert torch.isclose(data.data, data_2.data).all()
    assert torch.isclose(data.uncertainty, data_2.uncertainty).all()

Custom Transforms from :class:`~netloader.transforms.BaseTransform`
-------------------------------------------------------------------

The :class:`~netloader.transforms.BaseTransform` class provides the base functionality for all transforms.
When creating a custom transform from :class:`~netloader.transforms.BaseTransform`, if the transform's forwards or
backwards operations are different from an identity transform, then the ``forward`` and/or ``backward`` methods should
be defined.
If the transform's forwards or backwards uncertainty propagation is different from an identity transform, then the
``forward_grad`` and/or ``backward_grad`` methods should also be defined.
All custom transforms must also implement the ``__getstate__`` and ``__setstate__`` methods if extra attributes are
required.

Example implementing a logarithmic transform from :class:`~netloader.transforms.BaseTransform`:

.. code-block:: python

    from typing import Any
    from types import ModuleType

    import torch
    import numpy as np
    from netloader.utils.types import ArrayT
    from netloader.transforms import BaseTransform

    class Log(BaseTransform):
        """
        Logarithmic transform.
        """
        def __init__(self, *, base: float = 10) -> None:
            """
            Parameters
            ----------
            base : float, Optional
                Base of the logarithm, default = 10
            """
            super().__init__()
            self._base: float = base

        def __getstate__(self) -> dict[str, Any]:
            return super().__getstate__() | {'base': self._base}

        def __setstate__(self, state: dict[str, Any]) -> None:
            super().__setstate__(state)
            self._base = state['base']

        def forward(self, x: ArrayT) -> ArrayT:
            module: ModuleType = torch if isinstance(x, Tensor) else np
            return module.log(x) / np.log(self._base)

        def backward(self, x: ArrayT) -> ArrayT:
            return self._base ** x

        def forward_grad(self, x: ArrayT, uncertainty: ArrayT) -> tuple[ArrayT, ArrayT]:
            module: ModuleType = torch if isinstance(x, Tensor) else np
            return self(x), uncertainty / module.abs(x * np.log(self._base))

        def backward_grad(self, x: ArrayT, uncertainty: ArrayT) -> tuple[ArrayT, ArrayT]:
            module: ModuleType = torch if isinstance(x, Tensor) else np
            x = self(x, back=True)
            return x, uncertainty * module.abs(x * np.log(self._base))

Built-in Transforms
===================

All transforms currently supported by NetLoader are listed below with a brief description of their functionality and
their initialisation arguments, excluding general initialisation arguments defined in
:class:`~netloader.transforms.BaseTransform`.
For more information on each transform, see :ref:`API documentation <api_transforms>`.

.. autosig:: netloader.transforms
