.. _overview:

********
Overview
********

This page provides an overview of PyTorch Network Loader's architecture, design principles,
and core concepts.

What is PyTorch Network Loader?
===============================

PyTorch Network Loader is a utility package that simplifies the creation and training of neural
networks in PyTorch.
Instead of writing Python code and tracking input output shapes to define your network, you can use automatic &
dynamic shape tracking with built-in or custom layers defined in a JSON configuration file for simplicity or with Python
code for greater flexibility.
NetLoader also provides architecture harnesses for built-in training, inference, weight safe saving/loading, and other
supporting features for networks.
The main components of PyTorch Network Loader are:

* Simple & dynamic network creation
* Architectures for training and inference functionality
* Dataset & data loader creation
* Data transforms for preprocessing datasets

Design Principles
=================

Easy Configuration
------------------

Networks can be defined using JSON files, this makes it easy to:

* Experiment with different network designs quickly
* Share and version control network designs
* Be data agnostic

Shape Tracking
--------------

One of the most tedious aspects of creating networks in PyTorch is calculating input dimensions for each layer and
updating all layers if the input or output dimensions change.
NetLoader automatically tracks the output shape of each layer and uses it as the input for the next layer, eliminating
this manual step.

.. code-block:: json

    {
        "layers": [
            {"type": "Conv", "filters": 16},
            {"type": "Conv", "filters": 32},
            {"type": "Reshape", "shape": [-1]},
            {"type": "Linear", "factor": 1}
        ]
    }

We do not need to define the input shape, so the same network design can be used for different datasets, and the number
of features for the linear layer will be automatically calculated based on the output of the previous layer.
The output of the final linear layer will be the same as the output features of the dataset, so the same network design
can be used for different datasets with different output features.

Modularity
----------

The package is organized into distinct modules:

* ``layers`` - Individual layer implementations
* ``network`` - Network creation and shape tracking
* ``architectures`` - Network architectures (encoders, decoders, etc.)
* ``data`` - Dataset and data loader creation
* ``transforms`` - Data transforms for preprocessing datasets
* ``loss_funcs`` - Loss functions compatible with PyTorch safe weight loading
* ``schedulers`` - Schedulers to vary parameters during training, such as learning rate, with safe weight loading
* ``models`` - Pre-built model architectures (ConvNeXt, etc.)
* ``utils`` - Helper functions and utilities

Core Components
===============

Networks
--------

The :class:`~netloader.network.Network` class is the main entry point.
It:

* Loads JSON configurations
* Constructs the network
* Provides forward pass
* Handles network state saving & loading

.. code-block:: python

    import torch
    from netloader import Network

    # Define network
    net: Network = Network('encoder_config', './net_configs', in_shape=[3, 32, 32], out_shape=[10])

    # Forward pass
    output: torch.Tensor = net(torch.rand(1, 3, 32, 32))

    # Save network
    torch.save(net, 'encoder_net.pth')

Datasets & Data Loaders
-----------------------

The :class:`~netloader.data.datasets.BaseDataset` class is the base class for creating datasets.
It contains:

* High dimensional data
* Low dimensional data
* Extra meta data
* Unique IDs

The dataset can be combined with :func:`~netloader.data.datasets.loader_init` to create data loaders for training and testing.

.. code-block:: python

    from torch.utils.data import DataLoader
    from netloader.data import BaseDataset, loader_init

    # Define dataset
    class CustomDataset(BaseDataset):
        def __init__(self, data, labels):
            super().__init__()
            self.high_dim = data
            self.low_dim = labels

    # Create dataset
    dataset: CustomDataset = CustomDataset(data, labels)

    # Create train & validation data loaders
    data_loaders: tuple[DataLoader, DataLoader] = loader_init(dataset, ratios=(0.8, 0.2))

Data Transforms
---------------

The module :mod:`~netloader.transforms` provides data transforms for preprocessing datasets.
It:

* Supports PyTorch tensors and NumPy arrays
* Transforms and untransforms data
* Propagates uncertainties through data transformation

.. code-block:: python

    from netloader import transforms

    # Create transform
    transform: transforms.MultiTransform = transforms.MultiTransform(
        transforms.Normalise(data=dataset.high_dim),
        transforms.NumpyTensor(),
    )

    # Transform data
    dataset.high_dim = transform(dataset.high_dim)

Architectures & Training
------------------------

Architectures are the training harnesses for the networks, define how a network should be trained through the loss
function.
They include:

* Training & validation loop
* Inference
* Loss function definition
* Optimiser & scheduler initialisation
* Safe weight saving & loading

.. code-block:: python

    import numpy as np
    import netloader.architectures as archs

    # Create encoder architecture
    arch: archs.Encoder = archs.Encoder(
        1,
        './model_states',
        net,
        learning_rate=1e-4,
        in_transform=transform,
    )

    # Train network
    arch.training(100, data_loaders)

    # Inference
    predicts: dict[str, np.ndarray] = arch.predict(data_loaders[-1])

Next Steps
==========

* :doc:`../user_guide/network_configuration` - How to create networks
* :doc:`../user_guide/architectures` - How to train networks through architectures
* :doc:`../user_guide/data` - How to create datasets and data loaders
* :doc:`../user_guide/index` - For the full user guide on using the different components of NetLoader
* :doc:`../examples/examples` - For real-world examples using NetLoader
