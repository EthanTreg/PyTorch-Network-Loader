.. _install:

************
Installation
************

This page describes how to install PyTorch Network Loader for different use cases.

Requirements
============

PyTorch Network Loader requires:

* Python 3.12 or later
* PyTorch 2.11 or later
* NumPy
* Packaging

Optional Dependencies
---------------------

For normalising flows support (pip option flows):

* zuko

For Weights & Biases support (pip option wandb):

* wandb

For GPU acceleration (recommended):

* NVIDIA GPU with CUDA Toolkit

Using Within Projects
=====================

To use PyTorch Network Loader in your project, you can use pip and add it to your ``requirements.txt``:

.. code-block:: bash

    pip install netloader==<VERSION>

.. note::
   Replace ``<VERSION>`` with the specific version you want to use, or leave blank to install the latest version from
   the main branch.

With Optional Packages
----------------------

If you need optional support (currently flows and wandb), use:

.. code-block:: bash

    pip install netloader[option1,option2,...]

Local Development Installation
==============================

For local development or if you want to modify the package:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/EthanTreg/PyTorch-Network-Loader.git
       cd PyTorch-Network-Loader

2. Install dependencies (in a virtual environment):

   .. code-block:: bash

       pip install -r requirements.txt
