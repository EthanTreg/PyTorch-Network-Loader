.. _architectures:

*************
Architectures
*************

The :class:`~netloader.architectures.base.BaseArchitecture` class is a base class to provide the basic functionality for
training networks and that all network architectures inherit from.
PyTorch Network Loader provides some built-in architectures for common use cases, while more complex architectures can
inherit from :class:`~netloader.architectures.base.BaseArchitecture` and implement their custom logic.
:class:`~netloader.architectures.base.BaseArchitecture` was designed to train :class:`~netloader.network.Network`
objects (see :doc:`network_configuration`), but it can be used to train any PyTorch model through the
:class:`~netloader.network.CompatibleNetwork` class.

Basic Training
==============

The basic functionality of :class:`~netloader.architectures.base.BaseArchitecture` will be demonstrated using the
:class:`~netloader.architectures.encoder_decoder.Encoder` architecture.
Any class that inherits from :class:`~netloader.architectures.base.BaseArchitecture` will follow the same basic steps.

The first step is to initialise the dataset and data loaders (see :doc:`data` for more details on datasets and data
loaders):

.. code-block:: python

    from netloader.data import loader_init
    from torch.utils.data import DataLoader, Subset

    from src.data import CustomDataset

    dataset: CustomDataset = CustomDataset(...)
    loaders: tuple[DataLoader[Subset[CustomDataset]], DataLoader[Subset[CustomDataset]]] = \
        loader_init(dataset, batch_size=32)

Next, the network and architecture should be initialised (see :doc:`network_configuration` for more details on network
initialisation), in our case we are predicting a single value from (1,32,32) images.
In general, the architecture is initialised with the save name, directory to save the architecture, and the network to
train:

.. code-block:: python

    import torch
    from netloader.network import Network
    from netloader.architectures import Encoder

    # Create encoder network that predicts a value from 32x32 images
    net: Network = Network('net_name', 'config_dir/', [1, 32, 32], [1])

    # Initialise encoder architecture and move to device
    arch: Encoder = Encoder('save_name', 'states_dir/', net)
    arch.to('cuda' if torch.cuda.is_available() else 'cpu')

Finally, the architecture can be trained using the :meth:`~netloader.architectures.base.BaseArchitecture.training`
method, which takes the number of epochs that the network should be trained up to and the train and validation data
loaders.
The final predictions can be obtained using the :meth:`~netloader.architectures.base.BaseArchitecture.predict` method,
which takes the test data loader and returns the predictions:

.. code-block:: python

    from numpy import ndarray

    # Train the network up to 10 epochs
    arch.training(10, loaders)

    # Generate predictions on the test set
    predicts: dict[str, ndarray] = arch.predict(loaders[1])

The architecture by default automatically saves every epoch and can be loaded using :func:`~torch.load` or
:func:`~netloader.architectures.base.load_arch`.
To load the saved states in a PyTorch safe weights format, NetLoader needs to be imported:

.. code-block:: python

    from netloader.architectures import load_arch

    arch = load_arch('save_name', 'states_dir/', 'net_name', weights_only=True)

    # Or using torch.load, NetLoader is imported to add NetLoader classes to PyTorch safe globals
    import torch
    import netloader

    arch = torch.load('states_dir/net_name_save_name.pth', weights_only=True)

Features & Functionality of :class:`~netloader.architectures.base.BaseArchitecture`
===================================================================================

In addition to the basic training functionality, :class:`~netloader.architectures.base.BaseArchitecture` also contains
other useful features & functionality.

Extra Features
--------------

:class:`~netloader.architectures.base.BaseArchitecture` can be initialised with the arguments:

.. automethod:: netloader.architectures.base.BaseArchitecture.__init__
    :noindex:

If ``net`` is of type :class:`~torch.nn.Module` and not :class:`~netloader.network.Network` or
:class:`~netloader.network.CompatibleNetwork`, then the architecture will automatically wrap the network in a
:class:`~netloader.network.CompatibleNetwork` to make it compatible with the architecture's training and prediction
methods.

The attributes of the :class:`~netloader.architectures.base.BaseArchitecture` object include:

.. autoclass:: netloader.architectures.base.BaseArchitecture
    :no-members:
    :class-doc-from: class
    :no-index:

The ``losses`` attribute contains the training and validation losses for each epoch as either a float representing the
total loss or a dictionary containing the individual losses for each loss function and the total loss depending on the
architecture.
The method :meth:`~netloader.architectures.base.BaseArchitecture.get_losses` can be used to return the training and
validation losses as a dictionary containing a list of the losses for each epoch for each loss term.

The ``idxs`` attribute is used to contain the dataset IDs for the data used in the training data loader to keep track of
which data points were used for training and validation.

See :doc:`/api/architectures/base` for the methods of :class:`~netloader.architectures.base.BaseArchitecture` class.

Network Transforms
------------------

The :class:`~netloader.architectures.base.BaseArchitecture` class can be initialised with the dataset transforms and
saved to its state dictionary, which allows the transforms to be saved and loaded with the architecture.
The transforms should be applied to the data in the data loaders before being passed to
:meth:`~netloader.architectures.base.BaseArchitecture.training` or
:meth:`~netloader.architectures.base.BaseArchitecture.predict` methods.
The :meth:`~netloader.architectures.base.BaseArchitecture.predict` method will automatically apply the inverse
transforms to the predictions before returning them, so the predictions will be in the same format as the original data.

The transforms should be child classes of :class:`~netloader.transforms.BaseTransform`, see :doc:`/api/transforms`
for more details on transforms.

Optimisers & Schedulers
-----------------------

The optimiser and scheduler are defined in the :meth:`~netloader.architectures.utils.UtilityMixin.init_optimiser` and
:meth:`~netloader.architectures.utils.UtilityMixin.init_scheduler` methods respectively.
The :meth:`~netloader.architectures.utils.UtilityMixin.init_optimiser` method takes the parameter groups to optimise
and the argument ``optimiser_kwargs`` to initialise the architecture's optimiser.
The :meth:`~netloader.architectures.utils.UtilityMixin.init_scheduler` method takes the optimiser and the argument
``scheduler_kwargs`` to initialise the architecture's scheduler.
The parameter groups are defined in the :meth:`~netloader.architectures.base.BaseArchitecture.get_param_groups` method,
which can be overridden to define custom parameter groups and learning rates for the optimiser.
If a different optimiser or scheduler are required, the
:meth:`~netloader.architectures.utils.UtilityMixin.init_optimiser` and
:meth:`~netloader.architectures.utils.UtilityMixin.init_scheduler` methods can be overridden.

.. code-block:: python

    from typing import Any

    from torch import optim
    from netloader.architectures import Encoder


    class CustomArch(Encoder):
        def get_param_groups(self, learning_rate: tuple[float, float]) -> optim.optimizer.ParamsT:
            return [
                {'params': self.net[0].parameters(), 'lr': learning_rate[0]},
                {'params': self.net[1].parameters(), 'lr': learning_rate[1]},
            ]

        @staticmethod
        def init_optimiser(param_groups: optim.optimizer.ParamsT, **kwargs: Any) -> optim.Optimizer:
            return Adam(param_groups, **kwargs)

        @staticmethod
        def init_scheduler(
                optimiser: optim.Optimizer,
                **kwargs: Any) -> optim.lr_scheduler.LRScheduler:
            return StepLR(optimiser, **kwargs)


    # Or via monkey patching
    from typing import Any

    from torch import optim
    from netloader.architectures import Encoder


    def init_optimiser(param_groups: optim.optimizer.ParamsT, **kwargs: Any) -> optim.Optimizer:
        return Adam(param_groups, **kwargs)


    Encoder.init_optimiser = init_optimiser

If you change the scheduler, you may need to change the
:class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_update_scheduler`` method if the step to update the scheduler
requires any arguments (except for ReduceLROnPlateau which is automatically updated with the validation loss).

.. _loss-function-weights:

Loss Function Weights
---------------------

Depending on the architecture, there may be multiple loss functions used for training, requiring the use of loss weights
to weight the contribution of each loss function to the total loss.
The loss weights can be identified using the method
:meth:`~netloader.architectures.base.BaseArchitecture.get_loss_weights`, which returns a dictionary containing the loss
weights for each loss term or if a name is provided, the loss weight for that loss term.
The loss weights can be updated using the method
:meth:`~netloader.architectures.base.BaseArchitecture.set_loss_weights`, which takes either ``*args``, containing the
loss weights for each loss term in the same order as the loss terms returned by
:meth:`~netloader.architectures.base.BaseArchitecture.get_loss_weights`, or ``**kwargs``, containing the loss weights
for each loss term with the same names.

Random States & Reproduction
----------------------------

For reproducibility, the random number generator states for NumPy and PyTorch are saved in the architecture's state
dictionary.
First, to ensure reproducibility, a random seed should be set for NumPy and PyTorch along with disabling
non-deterministic algorithms in PyTorch:

.. code-block:: python

    import torch
    import numpy as np

    seed: int = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

Next, we also need to ensure that the data loader is deterministic by setting the generator and worker initialisation
function (assuming the data loader is using multiple workers):

.. code-block:: python

    import random

    from netloader.data import loader_init
    from torch.utils.data import DataLoader

    from src.data import CustomDataset


    def seed_worker(_: int) -> None:
        """
        Sets the random seed for data loader workers for reproducibility.
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataset: CustomDataset = CustomDataset(...)

    # Create data loader generator
    gen: Generator = Generator()
    gen.manual_seed(seed)

    # Create data loaders with generator and worker init function
    loaders: tuple[DataLoader[Subset[CustomDataset]], DataLoader[Subset[CustomDataset]]] = \
        loader_init(dataset, ..., generator=gen, worker_init_fn=seed_worker)

Finally, if continuing training from a saved architecture, when loading the architecture, it will set the random states
to the saved states.
Then, PyTorch will need to be set to use deterministic algorithms and the data loaders must set their random generator
states:

.. code-block:: python

    import torch

    from netloader.architectures import load_arch

    from src.networks import CustomArch

    arch: CustomArch = load_arch('save_name', 'states_dir/', 'net_name')

    # Set PyTorch to use deterministic algorithms
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Set data loader generator states to the saved architecture states
    loaders = loader_init(dataset, ..., generator=gen, worker_init_fn=seed_worker)
    loader_states: tuple[Tensor, Tensor] | None = arch.get_loader_states()

    for loader, state in zip(loaders, loader_states or []):
        loader.generator.set_state(state)

Weights & Biases Integration
----------------------------

There is built in support for `Weights & Biases (W&B) <https://wandb.ai/site>`_ logging in
:class:`~netloader.architectures.base.BaseArchitecture`.
Weights & Biases can be installed either via the optional pip installation of NetLoader with
``pip install netloader[wandb]`` or by installing W&B separately.

To use W&B logging, the :class:`~netloader.architectures.base.BaseArchitecture` attribute ``run`` must be set to a W&B
`Run <https://docs.wandb.ai/models/ref/python/public-api/runs#class-runs>`_ object.
The W&B `Run <https://docs.wandb.ai/models/ref/python/public-api/runs#class-runs>`_ object can be created with
the function `init <https://docs.wandb.ai/models/ref/python/functions/init#function-init>`_, see the
`W&B documentation <https://docs.wandb.ai/models/models_quickstart>`_ for more details.
:class:`~netloader.architectures.base.BaseArchitecture` will automatically log each training and validation losses for
each epoch to the `Run <https://docs.wandb.ai/models/ref/python/public-api/runs#class-runs>`_ object if it is defined.
Example usage:

.. code-block:: python

    import wandb
    from netloader.data import loader_init

    from src.data import CustomDataset
    from src.networks import CustomArch

    # Initialise data & architecture
    dataset: CustomDataset = CustomDataset(...)
    loaders: tuple[DataLoader[Subset[CustomDataset]], DataLoader[Subset[CustomDataset]]] = \
        loader_init(dataset, ...)
    arch: CustomArch = CustomArch(...)

    # Create W&B run
    arch.run = wandb.init(...)

    # Train architecture, losses will be automatically logged to W&B run
    arch.training(epochs, loaders)

    # Finish W&B run
    arch.run.finish()

Creating Custom Architectures
=============================

To create a custom architecture, first decide which architecture type to inherit from (see
:ref:`pre-defined-architectures` or :doc:`/api/architectures/index` for the available architecture types).

Then, depending on the requirements of your custom architecture, you may need to override some of the methods of the
base architecture class, such as:

- :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``__init__`` to add any additional arguments or attributes to the
  architecture.
- :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``__getstate__`` and
  :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``__setstate__`` to add any additional attributes to the state
  dictionary for saving and loading the architecture.
- :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_data_loader_translation`` to change if the input is
  high-dimensional data or low-dimensional data.
- :class:`~netloader.architectures.base.BaseArchitecture`\ .\ ``_loss_tensor`` to calculate the loss tensor for training and
  validation.
- :meth:`~netloader.architectures.base.BaseArchitecture.batch_predict` to define how the architecture makes predictions
  for a batch of data.
- :meth:`~netloader.architectures.base.BaseArchitecture.predict` to do any post-processing of the predictions.
- :meth:`~netloader.architectures.base.BaseArchitecture.to` to move any additional tensor attributes to the specified
  device and data type.

Creating Custom Encoder Architecture
------------------------------------

We can create an encoding architecture, one that uses regression or classification to predict low-dimensional data from
high-dimensional data, by first inheriting from the :class:`~netloader.architectures.base.BaseArchitecture` and defining
the ``__init__`` method.
In the ``__init__`` method, we can add the ``classes`` argument to specify the classes for classification, or if
``None``, the architecture will be used for regression:

.. code-block:: python

    from typing import Any

    from torch import nn, Tensor
    from netloader.architectures import BaseArchitecture
    from netloader.network import Network, CompatibleNetwork
    from netloader.loss_funcs import BaseLoss, MSELoss, CrossEntropyLoss


    class Encoder(BaseArchitecture):
        def __init__(
                self,
                save_num: int | str,
                states_dir: str,
                net: nn.Module | Network | CompatibleNetwork,
                *,
                classes: Tensor | None = None,
                **kwargs: Any) -> None:
            super().__init__(save_num, states_dir, net, **kwargs)
            self._loss_func_: BaseLoss
            self.classes: Tensor | None = classes

            # If classes are defined, use classification loss else regression
            if self.classes is None:
                self._loss_func_ = MSELoss()
            else:
                self.classes = self.classes.to(self.device)
                self._loss_func_ = CrossEntropyLoss()

Then, we need to define how the architecture can be saved and loaded with ``__getstate__`` and
``__setstate__`` methods, respectively:

.. code-block:: python

    class Encoder(BaseArchitecture):
        def __init__(...): ...

        def __getstate__(self) -> dict[str, Any]:
            # Save parent along with Encoder's classes & loss function
            return super().__getstate__() | {'classes': self.classes, 'loss_func': self._loss_func_}

        def __setstate__(self, state: dict[str, Any]) -> None:
            # Load parent state along with Encoder's classes & loss function
            super().__setstate__(state)
            self.classes = state['classes']
            self._loss_func_ = state['loss_func']

Next, we will define the loss function with the ``_loss_tensor`` method:

.. code-block:: python

    from netloader.utils import label_change
    from netloader.utils.types import TensorListLike


    class Encoder(BaseArchitecture):
        ...

        def _loss_tensor(self, in_data: TensorListLike, target: TensorListLike) -> Tensor:
            # Label one-hot encoding
            if self.classes is not None and isinstance(target, Tensor):
                target = label_change(target.squeeze(), self.classes)

            # Network forward pass
            outputs = self.net(in_data)

            # Compute & return loss
            return self._loss_func_(outputs, target)

Finally, if the network changes device, we need to also update the ``classes`` attribute to also change device:

.. code-block:: python

    from typing import Self


    class Encoder(BaseArchitecture):
        ...

        def to(self, *args: Any, **kwargs: Any) -> Self:
            super().to(*args, **kwargs)

            if self.classes is not None:
                self.classes = self.classes.to(self._device)
            return self

Creating Custom Autoencoder Architecture
----------------------------------------

We can create an autoencoder architecture, one where high-dimensional data is encoded to a low-dimensional latent
space, before being decoded back to a reconstructed version of the high-dimensional data.
For this problem, we can have multiple loss function terms, such as a regression reconstruction loss, a latent space
regression loss for physicalised latent spaces, and a
`Kullback–Leibler divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_ (KL) loss for
variational autoencoders, which can be weighted with the loss weights (see :ref:`loss-function-weights`).

We start the same way as for the encoder architecture, by inheriting from the
:class:`~netloader.architectures.base.BaseArchitecture` and defining the ``__init__`` method.
However, we also need to define the loss weights for the three loss functions and the transforms for the latent space.

.. code-block:: python

    from typing import Any

    from torch import nn
    from netloader.architectures import BaseArchitecture
    from netloader.transforms import BaseTransform
    from netloader.loss_funcs import BaseLoss, MSELoss
    from netloader.network import Network, CompatibleNetwork


    class Autoencoder(BaseArchitecture):
        def __init__(
                self,
                save_num: int | str,
                states_dir: str,
                net: nn.Module | Network | CompatibleNetwork,
                *,
                transform: list[BaseTransform] | BaseTransform | None = None,
                latent_transform: list[BaseTransform] | BaseTransform | None = None,
                **kwargs: Any) -> None:
            super().__init__(
                save_num,
                states_dir,
                net,
                transform=transform,
                in_transform=transform,  # Input & output transforms are the same for autoencoders
                **kwargs,
            )

            # Define loss functions, KL loss is calculated in Sample layers so not defined here
            self.reconstruction_func: BaseLoss = MSELoss()
            self.latent_func: BaseLoss = MSELoss()

            # Define loss weights
            self._loss_weights = {'reconstruct': 1, 'latent': 1e-2, 'kl': 1e-1}

            # Define latent space & target transforms
            self.transforms['latent'] = latent_transform
            self.transforms['targets'] = latent_transform

Then, we need to define how the architecture can be saved and loaded with ``__getstate__`` and
``__setstate__`` methods, respectively.
Loss weights & transforms are already saved by the parent class, so only the loss functions need to be saved.

.. code-block:: python

    class Autoencoder(BaseArchitecture):
        def __init__(...): ...

        def __getstate__(self) -> dict[str, Any]:
            return super().__getstate__() | {
                'reconstruction_func': self.reconstruction_func,
                'latent_func': self.latent_func,
            }

        def __setstate__(self, state: dict[str, Any]) -> None:
            super().__setstate__(state)
            self.reconstruction_func = state['reconstruction_func']
            self.latent_func = state['latent_func']

Next, we will define the loss functions with the ``_loss_tensor`` method:

.. code-block:: python

    from torch import Tensor
    from netloader.utils.types import TensorListLike


    class Autoencoder(BaseArchitecture):
        ...

        def _loss_tensor(
                self,
                in_data: TensorListLike,
                target: TensorListLike) -> dict[str, Tensor]:
            # Dictionary of losses, must have same keys as loss weights attribute
            loss: dict[str, Tensor] = {}
            latent: Tensor | None = None

            # Network forward pass
            output: Tensor = self.net(in_data)

            # Get latent space from last checkpoint
            if self.net.checkpoints:
                latent = self.net.checkpoints[-1]

            # Reconstruction loss
            if self.get_loss_weights('reconstruct'):
                loss['reconstruct'] = self.reconstruction_func(output, in_data)

            # Latent space loss
            if self.get_loss_weights('latent') and latent is not None:
                loss['latent'] = self.latent_func(latent, target)

            # KL loss comes from Network.kl_loss attribute if there is a Sample layer
            if self.get_loss_weights('kl'):
                loss['kl'] = self.net.kl_loss

            # Total loss is automatically calculated as the weighted sum from loss weights
            return loss

Finally, we want to return both the reconstructions and the latent space when generating predictions as
:class:`~netloader.architectures.base.BaseArchitecture` only returns the forward pass output by default.
Therefore, we need to define the ``batch_predict`` method.
``batch_predict`` must return an :type:`~netloader.utils.types.NDArrayLike` or ``None`` for each transform in the
``transforms`` attribute (excluding the first three keys of ``ids``, ``inputs``, and ``targets``), so by default
``preds``, the predictions, plus the two additional keys of ``latent`` and

.. code-block:: python

    from netloader.utils.types import NDArrayListLike


    class Autoencoder(BaseArchitecture):
        ...

        def batch_predict(
                self,
                data: TensorListLike,
                **_: Any) -> tuple[NDArrayListLike | None, ...]:
            output: NDArrayListLike = self.net(data).detach().cpu().numpy()
            return output

.. _pre-defined-architectures:

Pre-Defined Architectures
=========================

All architectures currently supported by NetLoader are listed below with a brief description of their functionality and
their initialisation arguments, excluding general initialisation arguments defined in
:class:`~netloader.architectures.base.BaseArchitecture`.
For more details on each layer, please refer to the :ref:`API documentation <api_architectures>`.

Encoder-Decoder
---------------

Encoder-decoder architectures are the standard regression/classification architectures.
See :doc:`/api/architectures/encoder_decoder` for more details.

.. autoarchs:: netloader.architectures.encoder_decoder.Autoencoder

.. autoarchs:: netloader.architectures.encoder_decoder.Decoder

.. autoarchs:: netloader.architectures.encoder_decoder.Encoder

Flows
-----

Flow architectures are used to train normalising flows.
See :doc:`/api/architectures/flows` for more details.

.. autoarchs:: netloader.architectures.flows
