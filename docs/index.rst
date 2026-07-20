.. _netloader-docs:
.. meta::
   :google-site-verification: L3-TegeT7VmexU4N80NyedPKBKxd4-5mpG5ouL0h-K8

***************************************
PyTorch Network Loader Documentation
***************************************

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg
   :target: https://pytorch.org/
   :alt: PyTorch Version

**Version**: |release|

**Useful links:**
`Source Repository <https://github.com/EthanTreg/PyTorch-Network-Loader>`_ |
`Issue Tracker <https://github.com/EthanTreg/PyTorch-Network-Loader/issues>`_ |
`Real-World Example <https://github.com/EthanTreg/Bayesian-DARKSKIES>`_

PyTorch Network Loader is a utility that enables the easy creation of neural networks in PyTorch using JSON
configuration files.
The package automatically tracks layer output shapes, eliminating the need to manually calculate input dimensions for
each layer.
NetLoader also provides an architecture framework that allows for the easy architecture setup and training framework,
along with additional useful utilities to accelerate the use of neural networks in science.

Key Features
============

* **JSON-Based Network Definition**: Define neural networks using simple JSON files
* **Automatic Shape Tracking**: No need to calculate input shapes for each layer
* **Built-in Training Pipeline**: Architectures come with all training functionality included
* **Flexible Architecture Support**: Support for pre-built and custom architectures, such as encoders, autoencoders, and
  normalizing flows
* **Datasets & Data Loaders Framework**: Easily create datasets and data loaders for training and evaluation, along with
  modular data transforms

Quick Example
=============

.. code-block:: python

    import netloader.architectures as archs
    from netloader.network import Network

    # Load network from JSON configuration
    net = Network('config.json', '/path/to/configs/', in_shape=[3, 224, 224], out_shape=[10])

    # Create architecture
    arch = archs.Encoder(1, '/path/to/save/states/', net, learning_rate=1e-3)

    # Train the model for 100 epochs
    arch.training(100, (train_loader, val_loader))

    # Make predictions
    predictions = arch.predict(test_loader)

Getting Started
===============

.. grid:: 1 2 2 2
    :gutter: 2

    .. grid-item-card:: Installation
        :link: getting_started/install
        :link-type: doc

        Installation instructions

    .. grid-item-card:: User Guide
        :link: user_guide/index
        :link-type: doc

        Guides on using the different components

    .. grid-item-card:: Examples
        :link: examples/examples
        :link-type: doc

        Real-world examples using NetLoader

    .. grid-item-card:: API Reference
        :link: api/index
        :link-type: doc

        API documentation for all modules

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   getting_started/install
   getting_started/overview

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   examples/examples

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Development

   development/changelog

