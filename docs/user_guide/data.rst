.. _data:

***********************
Datasets & Data Formats
***********************

PyTorch Network Loader provides the :class:`~netloader.data.datasets.BaseDataset` class for creating custom datasets in
a format compatible with NetLoader architectures, along with additional useful classes and functions for data
formatting.

Data Requirements for NetLoader Architectures
=============================================

NetLoader follows the standard PyTorch dataset conventions (see the
`PyTorch Datasets <https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ tutorial for more
information), but with some specific requirements for compatibility with NetLoader architectures.
The basic requirements for a custom dataset are:

- ``__len__`` method that returns the number of samples in the dataset.
- ``__getitem__`` method that returns a tuple of: ID, low-dimensional data (or None), high-dimensional data, and extra
  (or None) for a single sample.

High-dimensional data should always be set, if there is only one set of data (such as in unsupervised learning) then the
low-dimensional data can be set to None.
The ID is a unique identifier for each sample which can be used for tracking samples in the training and validation data
loaders.
The extra field can be used to store any additional information about the sample that can be used in the loss function
of the architecture.

Custom Datasets from :class:`~netloader.data.datasets.BaseDataset`
------------------------------------------------------------------

:class:`~netloader.data.datasets.BaseDataset` provides a convenient base class for creating custom datasets compatible
with :class:`~netloader.architectures.base.BaseArchitecture` architectures.

The attributes of the :class:`~netloader.data.datasets.BaseDataset` object include:

.. autoclass:: netloader.data.datasets.BaseDataset
    :no-members:
    :class-doc-from: class
    :no-index:

The minimum requirement to use :class:`~netloader.data.datasets.BaseDataset` is to inherit
:class:`~netloader.data.datasets.BaseDataset` and define the attribute ``high_dim``, which if an architecture only uses
one set of data, it should use the ``high_dim`` attribute.
However, depending on the architecture, the dataset may also need to define the ``low_dim`` and ``extra`` attributes.
The ``idxs`` attribute is set to a list of indices for the samples in the dataset by default, but can be overridden if
needed.

The :class:`~netloader.data.datasets.BaseDataset` class also has the following methods:

.. automethod:: netloader.data.datasets.BaseDataset.get_extra
    :noindex:

.. automethod:: netloader.data.datasets.BaseDataset.get_high_dim
    :noindex:

.. automethod:: netloader.data.datasets.BaseDataset.get_low_dim
    :noindex:

.. automethod:: netloader.data.datasets.BaseDataset.step
    :noindex:

Along with the magic methods:

.. automethod:: netloader.data.datasets.BaseDataset.__getitem__
    :noindex:

.. automethod:: netloader.data.datasets.BaseDataset.__len__
    :noindex:

If pre-processing of the low-dimensional, high-dimensional, or extra data is needed for each time a sample is accessed
(such as data augmentation), the methods ``get_low_dim``, ``get_high_dim``, and ``get_extra`` can be overridden to
perform the necessary pre-processing, static pre-processing should be applied to the saved data, in the ``__init__``
method, or after the dataset has been initialised and transformed with transforms from :mod:`~netloader.transforms`
(see :doc:`transforms` for more information).

To create a custom dataset, inherit from :class:`~netloader.data.datasets.BaseDataset` and define the necessary
attributes and methods based on the requirements of the architecture.
For example, if using the :class:`~netloader.architectures.encoder_decoder.Encoder` architecture, the dataset should
define the ``low_dim`` attribute as the labels and the ``high_dim`` attribute as the input data:

.. code-block:: python

    import pickle

    import torch
    from torchvision.transforms import v2
    from netloader.data import BaseDataset

    class CustomDataset(BaseDataset):
        def __init__(self, data_dir: str) -> None:
            super().__init__()

            with open(data_dir, 'rb') as file:
                self.low_dim, self.high_dim = pickle.load(file)  # e.g. labels & images

            # Data augmentations
            self.aug = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
            ])

        def get_high_dim(self, idx: int) -> torch.Tensor:
            # Apply augmentations to the high-dimensional data
            return self.aug(self.high_dim[idx])

If any parameters in the dataset should be updated during training, such as the level of noise induced in the dataset,
the method ``step`` can be implemented to update the necessary parameters at the end of each batch iteration during the
training of the architecture:

.. code-block:: python

    from multiprocessing import Value
    from multiprocessing.sharedctypes import Synchronized

    class CustomDataset(BaseDataset):
        max_epochs: int = 100
        max_noise: float = 0.1

        def __init__(self, data_dir: str) -> None:
            super().__init__()
            # Shared value for the noise level so that all workers in the data loader can access and
            # update it
            self.noise: Synchronized[float] = Value('f', 0.0)

            with open(data_dir, 'rb') as file:
                self.low_dim, self.high_dim = pickle.load(file)

        def step(self, epoch: float) -> None:
            # Update the noise level at the end of each batch iteration
            self.noise.value = min(self.max_noise, self.max_noise * epoch / self.max_epochs)

        def get_high_dim(self, idx: int) -> torch.Tensor:
            # Add noise to the high-dimensional data
            return self.high_dim[idx] + torch.randn_like(high_dim_data) * self.noise_level.value

Data with Uncertainties
=======================

If the dataset contains uncertainties, the data can be stored as a :class:`~netloader.data.structures.Data` object,
which contains the data and the corresponding uncertainties that can be naturally transformed using
:mod:`~netloader.transforms` (see :doc:`transforms` for more information) and passed to the architectures; however,
currently the architectures do not fully support the use of :class:`~netloader.data.structures.Data` objects and will
instead concatenate the data and uncertainties as a :class:`~torch.Tensor`.

The :class:`~netloader.data.structures.Data` class can be initialised with the following parameters:

.. automethod:: netloader.data.structures.Data.__init__
    :noindex:

The attributes of the :class:`~netloader.data.datasets.BaseDataset` object include:

.. autoclass:: netloader.data.structures.Data
    :no-members:
    :class-doc-from: class
    :no-index:

See :doc:`/api/data/structures` for the methods of :class:`~netloader.data.structures.Data` class.

.. code-block:: python

    import torch
    from netloader.data import Data
    from netloader import transforms

    values: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
    uncertainties: torch.Tensor = torch.tensor([0.1, 0.2, 0.3])
    data: Data = Data(values, uncertainty=uncertainties)
    trans_data: Data = transforms.Log()(data)

Multiple Inputs/Outputs
=======================

If the architecture has multiple inputs or outputs, the dataset can store the multi-data as a
:class:`~netloader.data.structures.DataList` object.
:class:`~netloader.data.structures.DataList` objects can store :class:`~numpy.ndarray`, :class:`~torch.Tensor`, or
:class:`~netloader.data.structures.Data` objects where each array or :class:`~netloader.data.structures.Data` object has
the same number of samples (i.e. the same length along the first dimension).
Some network layers from :mod:`~netloader.layers` support :class:`~netloader.data.structures.DataList` objects as inputs
and/or outputs, such as :class:`~netloader.layers.multi_layer.Unpack` (see :doc:`/api/layers/index` for more
information).

The :class:`~netloader.data.structures.DataList` class can be initialised with the following parameters:

.. automethod:: netloader.data.structures.DataList.__init__
    :noindex:

The :class:`~netloader.data.structures.DataList` class also has the following magic methods:

.. automethod:: netloader.data.structures.DataList.__getitem__
    :noindex:

.. automethod:: netloader.data.structures.DataList.__iter__
    :noindex:

.. automethod:: netloader.data.structures.DataList.__len__
    :noindex:

Along with matching methods with more functionality:

.. automethod:: netloader.data.structures.DataList.get
    :noindex:

.. automethod:: netloader.data.structures.DataList.iter
    :noindex:

.. automethod:: netloader.data.structures.DataList.len
    :noindex:

The methods above have the ``list_`` argument, which specifies if the method should act on the list containing the data,
or the data itself.
For example, ``DataList.len(True)`` will return the number of data arrays in the list, while ``DataList.len(False)``
will return the number of samples in each data array.
For the magic methods, the ``list_`` argument cannot be specified, so the default behaviour follows:

- ``__getitem__``: ``list_=False``, returns a DataList with each data array indexed at the specified index.
- ``__iter__``: ``list_=True``, iterates over the data arrays in the list.
- ``__len__``: ``list_=False``, returns the number of samples in each data array.

See :doc:`/api/data/structures` for the methods of :class:`~netloader.data.structures.DataList` class.

.. code-block:: python

    import torch
    import numpy as np
    from netloader.data import Data, DataList

    data1: np.ndarray = np.random.rand(4, 3, 10)
    data2: torch.Tensor = torch.arange(12).view(4, 3)
    data3: Data = Data(torch.ones((4, 3)), uncertainty=torch.zeros((4, 3)))
    data_list: DataList = DataList([data1, data2, data3])

    assert len(data_list) == 4
    assert [tuple(data.shape) for data in data_list] == [(4, 3, 10), (4, 3), (4, 3)]
    assert [tuple(data.shape) for data in data_list[0]] == [(3, 10), (3,), (3,)]

Data List Naming
----------------

The :class:`~netloader.data.structures.DataList` class can also have names assigned to each data array in the list,
which can be used to access the data arrays by name instead of by index.
The names can be used to convert a :class:`dict` of data arrays to a :class:`~netloader.data.structures.DataList` object
or vice versa.
The methods :meth:`~netloader.data.structures.DataList.get_names` and
:meth:`~netloader.data.structures.DataList.set_names` can be used to get and set the names of the data arrays in the
list, respectively.

.. code-block:: python

    import torch
    import numpy as np
    from netloader.data import Data, DataList

    data_dict: dict
    data1: np.ndarray = np.random.rand(4, 3, 10)
    data2: torch.Tensor = torch.arange(12).view(4, 3)
    data3: Data = Data(torch.ones((4, 3)), uncertainty=torch.zeros((4, 3)))
    data_list2: DataList
    data_list1: DataList = DataList([data1, data2, data3], names=['data1', 'data2', 'data3'])

    data_list1.set_names(['new_data1', 'new_data2', 'new_data3'])
    data_dict = data_list1.get_data(True)
    data_list2 = DataList(data_dict)

    assert data_list1 == data_list2
    assert isinstance(data_dict, dict)
    assert data_list1.get_names() == ['new_data1', 'new_data2', 'new_data3']
    assert data_list1['new_data1'].shape == (4, 3, 10)
