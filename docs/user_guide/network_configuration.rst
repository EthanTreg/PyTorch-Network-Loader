.. _network_configuration:

*********************
Network Configuration
*********************

The :class:`~netloader.network.Network` class enables the easy creation of a PyTorch neural network from ``.json`` files
and without requiring knowledge of the output shape of each layer, and therefore, easy modification to inputs and/or
layers.
There are two main things required to make a PyTorch neural network from ``.json`` files:

1. Define the layers of the network in a ``.json`` file
2. Build the network object using the :class:`~netloader.network.Network` class

Configuration File Structure
============================

The file structure follow:

- ``net``: ``dict[str, Any]`` - Global network parameters and metadata following
    - ``checkpoints``: ``bool = False`` - Whether to capture intermediate outputs as checkpoints (uses more memory)
    - ``paper``: ``str = ''`` - Reference to the paper describing the network
    - ``github``: ``str = ''`` - Link to GitHub repository for the network
    - ``description``: ``str = ''`` - Description of the network
    - *<layer_name>*: ``dict[str, Any]`` - Default parameters for specific layer given by *<layer_name>* that overrides
      the default parameters for that type of layer
- ``layers``: ``list[dict[str, Any]]`` - Dictionaries containing information on each layer, where the first layer takes
  the input, and the last layer produces the output

Examples of the layers can be found under the section :doc:`layers` and examples of ``.json`` files can be found in the
directory ``network_configs``.

Configuration File Example
--------------------------

If we want to make a simple encoder-type architecture composed of three linear layers, we can create the following
``.json`` file:

.. code-block:: json

    {
      "net": {
        "checkpoints": true,
        "Linear": {
          "dropout": 0.1
        }
      },
      "layers": [
        {
          "type": "Linear",
          "features": 120
        },
        {
          "type": "Linear",
          "features": 120
        },
        {
          "type": "Linear",
          "factor": 1,
          "activation": null
        }
      ]
    }

First, we set ``checkpoints`` to ``true`` in the ``net`` dictionary, which means that the output of each layer will not
be automatically cached and instead only be cached when a :class:`~netloader.layers.misc.Checkpoint` is used.
Since we don't need to reuse the output of any layer, we can save the RAM usage by setting this to ``true``.
We then set the default dropout for all linear layers to 0.1.

In the layers list, we specify three linear layers, where the first two have 120 output features and the last layer is
agnostic to the output shape by setting ``factor`` to 1, which means the output shape will be the same as the output
shape given to :class:`~netloader.network.Network` when building the network.
We don't need to specify the input shape for the first layer, or the output shape for any of the layers, because when we
build the network using the :class:`~netloader.network.Network` class, it will automatically track the shape of the
output of each layer.

Building the Network
====================

To build the network from the configuration file, we can use the :class:`~netloader.network.Network` class, which can be
initialised with the following parameters:

.. automethod:: netloader.network.Network.__init__
    :noindex:

.. code-block:: python

    from netloader.network import Network

    # Load the network configuration from the JSON file
    net = Network(
        'network_config_name',  # Name of JSON file, .json extension is optional
        'relative/path/to/network/configs/directory/',
        [240],  # Input shape (excluding batch dimension)
        [1],  # Output shape (excluding batch dimension)
        root='/absolute/path/to/project/directory/',  # Not required
    )

Then the :class:`~netloader.network.Network` object can be used as a standard PyTorch :class:`~torch.nn.Module`:

.. code-block:: python

    import torch

    input_data = torch.ones((8, 240))  # Batch of 8 samples with input shape (240)
    target = torch.ones((8, 1))  # Batch of 8 samples with target shape (1)
    optimiser = torch.optim.AdamW(net.parameters(), lr=0.001)

    # Network forward pass
    output = net(input_data)

    # Compute loss and backpropagate
    loss = torch.nn.MSELoss()(output, target)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    torch.save(net, '/path/to/save/model.pth')

    # Loading the model requires first importing netloader for loading with safe weights
    import torch
    import netloader
    net = torch.load('/path/to/save/model.pth', weights_only=True)


Extra :class:`~netloader.network.Network` Features
--------------------------------------------------

The attributes of the :class:`~netloader.network.Network` object include:

.. autoclass:: netloader.network.Network
    :no-members:
    :class-doc-from: class
    :no-index:

The ``group`` attribute is an integer that only activates layers that have the same group number or a group number of 0
(default).
The ``layer_num`` attribute activates all layers up to the specified layer number (negative numbers count from the end).
The ``checkpoints`` attribute is a list of the outputs of the layers that have been captured using the
:class:`~netloader.layers.misc.Checkpoint` layer, which can be accessed after the forward pass.
The ``kl_loss`` attribute is the KL divergence loss set if there is a :class:`~netloader.layers.linear.Sample` layer in
the network.

The :class:`~netloader.network.Network` class also has the following methods:

.. automethod:: netloader.network.Network.get_config
    :noindex:

.. automethod:: netloader.network.Network.forward
    :noindex:

.. automethod:: netloader.network.Network.to
    :noindex:

Along with the magic methods:

.. automethod:: netloader.network.Network.__getitem__
    :noindex:

.. automethod:: netloader.network.Network.__iter__
    :noindex:

.. automethod:: netloader.network.Network.__len__
    :noindex:

Building a Network Using Python
===============================

If you would like more functionality than what is available in the ``.json`` configuration files, you can extend the
class :class:`~netloader.network.Network` and override the ``__init__`` method to build the network using layers from
:mod:`~netloader.layers` and assigning the ``net`` attribute to a :class:`~torch.nn.ModuleList` of the layers.
You will also need to override the ``__setstate__`` & ``__getstate__`` magic methods to ensure the network can be saved
and loaded properly.
The forward method is built into the :class:`~netloader.network.Network` class, expecting a sequential
:class:`~torch.nn.ModuleList` of :class:`~netloader.layers.base.BaseLayer` objects.

.. code-block:: python

    from netloader import layers
    from netloader.utils import Config
    from netloader.network import Network

    class CustomNetwork(Network):
        def __init__(
                self,
                name: str,
                in_shape: list[int],
                out_shape: list[int],
                hidden_features: list[int]) -> None:
            # Initialise an empty network
            super().__init__(
                name,
                Config(),
                in_shape,
                [],
                suppress_warning=True,
            )
            self.in_shape: list[int] = in_shape
            self.out_shape: list[int] = out_shape
            self.hidden_features: list[int] = hidden_features
            self.build_net(self.in_shape, self.out_shape, self.hidden_features)

        # Required for saving
        def __getstate__(self) -> dict[str, Any]:
            return super().__getstate__() | {
                'in_shape': self.in_shape,
                'out_shape': self.out_shape,
                'hidden_features': self.hidden_features,
            }

        # Required for loading
        def __setstate__(self, state: dict[str, Any]) -> None:
            super().__init__(
                state['name'],
                Config(),
                state['shapes'][0],
                [],
                suppress_warning=True,
            )
            self.in_shape = state['in_shape']
            self.out_shape = state['out_shape']
            self.hidden_features = state['hidden_features']
            self.build_net(self.in_shape, self.out_shape, self.hidden_features)

        def build_net(
                self,
                in_shape: list[int],
                out_shape: list[int],
                hidden_features: list[int]) -> None:
            self.net.extend([layers.Linear(
                out_shape,
                self.shapes,
                features=feature,
                activation='SELU',
            ) for feature in hidden_features])
            self.net.append(layers.Linear(out_shape, self.shapes, factor=1, activation=None))

If you want to load a custom network using PyTorch safe weights, you will need to add your custom network to PyTorch's
safe globals.
NetLoader does this automatically for any child classes of a NetLoader class if they have already been initialised.
Otherwise, you can add your custom network to the safe globals manually before loading the saved model:

.. code-block:: python

    from netloader.utils import safe_globals

    # Your module containing the custom network class
    from custom_package import models

    # All classes in the models module belonging to 'custom_package' will be added to the safe globals
    safe_globals('custom_package', [models])
    net = torch.load('/path/to/saved/model.pth', weights_only=True)

See :class:`~netloader.models.convnext.ConvNeXt` for an example of a custom network built using Python.

Combining Multiple Networks
===========================

If you want to combine multiple networks, such as for an autoencoder (where there is an encoder and a decoder), you can
create separate ``.json`` files for each network, then the class MultiNetwork will combine them into a single network:

.. code-block:: python

    import torch
    from netloader.models import MultiNetwork

    net = MultiNetwork(
        'autoencoder',
        'encoder',
        'decoder',
        root='/absolute/path/to/project/directory/',
        config='relative/path/to/network/configs/directory/',
        in_shape=[3, 64, 64],
        out_shapes=[[128], [3, 64, 64]],
    )
    reconstruction = net(torch.ones((8, 3, 64, 64)))  # Full autoencoder forward pass
    latent = net[0](torch.ones((8, 3, 64, 64))))  # Encoder only forward pass
    reconstruction = net[1](latent)  # Decoder only forward pass

Converting PyTorch Networks to :class:`~netloader.network.Network` Compatible (Experimental)
============================================================================================

If you have an existing PyTorch network (child of :class:`~torch.nn.Module`) and want to use it with NetLoader
architectures, you can convert it to be in a compatible format by either inheriting
:class:`~netloader.network.CompatibleNetwork` or by passing the PyTorch network to the
:class:`~netloader.network.CompatibleNetwork`:

.. code-block:: python

    import torch
    from torchvision.models import resnet18
    from netloader.network import CompatibleNetwork

    # Inheriting from CompatibleNetwork
    class CustomNetwork(CompatibleNetwork):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__(torch.nn.Sequential(
                torch.nn.Linear(in_features, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, out_features),
            ))

    net = CustomNetwork(240, 1)
    output = net(torch.ones((1, 240)))

    # Converting existing PyTorch network to CompatibleNetwork
    net = CompatibleNetwork(resnet18())
    output = net(torch.ones((1, 3, 224, 224)))

However, loading saved networks using :class:`~netloader.network.CompatibleNetwork` is not guaranteed to be compatible
with PyTorch safe weights.
