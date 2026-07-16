.. _layers:

******
Layers
******

PyTorch Network Loader provides several built-in layers that can be used to construct neural networks.
They are built upon :class:`~netloader.layers.base.BaseLayer` and utilise :class:`~netloader.utils.utils.Shapes` for
layer output shape tracking and input/output shape flexibility, alongside PyTorch safe weight saving.

See the configuration files under `network_configs <https://github.com/EthanTreg/PyTorch-Network-Loader/tree/master/network_configs>`_
for examples on how to use layers in ``.json`` files.
See models under `netloader.models <https://github.com/EthanTreg/PyTorch-Network-Loader/tree/master/netloader/models>`_
for examples on how to use layers in Python.

Layer Shape Compatibility
=========================

Linear layers take inputs of :math:`(N,\ldots,L)` where :math:`N` is the batch size and :math:`L` is the length of the
input.
Recurrent layers require an input shape of :math:`(N,C,L)`, where :math:`C` is the number of channels/sequence length.
Some layers, such as convolutional, require dimension :math:`C`, but can take 1D, 2D, & 3D inputs, so the inputs would
have shapes :math:`(N,C,L)`, :math:`(N,C,H,W)`, or :math:`(N,C,D,H,W)`, respectively, where :math:`D` is the depth,
:math:`H` is the height, :math:`W` is the width, and :math:`L` is the length for 1D data.

The :class:`~netloader.layers.misc.Reshape` layer can be used to change the shape of the inputs for compatibility
between the layers.

Extra Layer Features
====================

Features of :class:`~netloader.layers.base.BaseLayer`
-----------------------------------------------------

All layers inherit from :class:`~netloader.layers.base.BaseLayer` which has the signature:

.. automethod:: netloader.network.Network.__init__
    :noindex:

All layers can pass arguments to :class:`~netloader.layers.base.BaseLayer` through their ``**kwargs`` argument.
If an argument is passed to :class:`~netloader.layers.base.BaseLayer` that is not recognised, it will log a warning and
ignore the argument.

The :class:`~netloader.layers.base.BaseLayer` class has the following attributes that are available to all layers:

.. autoclass:: netloader.layers.base.BaseLayer
    :no-members:
    :class-doc-from: class
    :no-index:

The ``group`` argument means that the layer will only be active if the network attribute ``group`` is equal to the
layer's ``group``.
This is most useful if the network has multiple heads that get changed during training.
:class:`~netloader.layers.multi_layer.Skip` layers take the output from a specified layer so should be used between
groups so that all ``group`` paths have the same input.
See `layer_examples.json <https://github.com/EthanTreg/PyTorch-Network-Loader/blob/master/network_configs/layer_examples.json>`_
for an example.

The :class:`~netloader.layers.base.BaseLayer` class has the following methods:

.. automethod:: netloader.layers.base.BaseLayer.forward
    :noindex:

.. automethod:: netloader.layers.base.BaseLayer.to
    :noindex:

Along with the magic methods:

.. automethod:: netloader.layers.base.BaseLayer.__getstate__
    :noindex:

You can also implement the ``extra_repr`` method to add extra information to the layer's string representation.
If inheriting from either :class:`netloader.layers.base.BaseLayer` or :class:`netloader.layers.base.BaseSingleLayer`,
then the ``extra_repr`` method is not implemented, so the custom ``extra_repr`` should not call ``super``, while for
:class:`netloader.layers.base.BaseMultiLayer`, the ``extra_repr`` method is implemented so ``super`` should be called.

Features of :class:`~netloader.layers.base.BaseSingleLayer`
-----------------------------------------------------------

Some layers utilise a simple sequential layer structure such as :class:`~netloader.layers.linear.Linear` &
:class:`~netloader.layers.convolutional.Conv`, and so inherit from :class:`~netloader.layers.base.BaseSingleLayer` which
has the signature:

.. automethod:: netloader.layers.base.BaseSingleLayer.__init__
    :noindex:

The :class:`~netloader.layers.base.BaseSingleLayer` class has the following attributes:

.. autoclass:: netloader.layers.base.BaseSingleLayer
    :no-members:
    :class-doc-from: class
    :no-index:

The :class:`~netloader.layers.base.BaseSingleLayer` class has the following methods:

.. automethod:: netloader.layers.base.BaseSingleLayer.forward
    :noindex:

Features of :class:`~netloader.layers.base.BaseMultiLayer`
----------------------------------------------------------

Some layers utilise the outputs from other layers such as :class:`~netloader.layers.multi_layer.Skip` &
:class:`~netloader.layers.multi_layer.Concatenate`, and so inherit from :class:`~netloader.layers.base.BaseMultiLayer`
which has the signature:

.. automethod:: netloader.layers.base.BaseMultiLayer.__init__
    :noindex:

Creating Custom Layers
======================

To create a custom layer, first decide which base layer to inherit from:

- :class:`~netloader.layers.base.BaseSingleLayer` - Layers that can use a simple sequential forward pass.
- :class:`~netloader.layers.base.BaseMultiLayer` - Layers that require the outputs from other layers.
- :class:`~netloader.layers.base.BaseLayer` - Layers that require more complex functionality.

All layers ``__init__`` method must have a ``shapes`` and a ``**kwargs`` parameter, and the ``**kwargs`` parameter must
be passed to the base layer's ``__init__`` method.
The ``shapes`` parameter is used to track the output shapes for each layer (excluding batch dimension); therefore, the
output shape of the custom layer must be appended to the ``shapes`` parameter in the ``__init__`` method (even if the
shape remains unchanged).
They must also implement the ``__getstate__`` method calling the base layer's ``__getstate__`` method and returning a
dictionary of the layer's custom ``__init__`` parameters.

Custom layers can also use the following ``__init__`` parameters:

- ``net_check`` : :class:`bool` - Global flag to show if the network saves the outputs from each layer (False) or only
  from :class:`~netloader.layers.misc.Checkpoint` layers (True), default = False.
- ``idx`` : :class:`int` - The index of the layer in the network, if used it must be passed to the base layer's
  ``__init__`` method, default = 0.
- ``root`` : :class:`str` - The root directory for the project, used for relative file paths, default = ''.
- ``version`` : :class:`str` - The version of NetLoader, used for backwards compatibility, if used it must be passed to
  the base layer's ``__init__`` method, default = ``netloader.__version__``.
- ``net_out`` : :class:`list`\[:class:`int`\] - Output shape for the whole network.
- ``check_shapes`` : :class:`~netloader.utils.utils.Shapes` - The output shape for each checkpoint layer.

The forward method must include the input ``x`` with type :class:`~netloader.utils.types.TensorListLike` and the
``*args`` & ``**kwargs`` parameters in the ``forward`` method.
Custom layers can use the following ``forward`` method parameters:

- ``outputs`` : :class:`list`\[:class:`~torch.Tensor`\] - The outputs from each layer in the network if the network is not
  exclusively using checkpoint outputs.
- ``checkpoints`` : :class:`list`\[:class:`~torch.Tensor`\] - The outputs from each checkpoint layer in the network.
- ``net`` : :class:`~typing.Self` - The network that the layer is in, used for accessing the network's
  attributes.

Custom Layer From :class:`~netloader.layers.base.BaseSingleLayer`
-----------------------------------------------------------------

If your custom layer can use :class:`~torch.nn.Sequential` for the forward pass, then you can inherit from
:class:`~netloader.layers.base.BaseSingleLayer` and append to the ``layers`` attribute in the ``__init__`` method.

.. code-block:: python

    from typing import Any

    from torch import nn
    from netloader.utils import Shapes
    from netloader.layers.base import BaseSingleLayer

    class CustomLinear(BaseSingleLayer):
        def __init__(
                self,
                features: int,
                activation: str | None,
                shapes: Shapes,
                **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._features: int = features
            in_shape: list[int] = shapes[-1].copy()
            self.layers.add_module('Linear', nn.Linear(in_shape[-1], self._features))

            if activation:
                self.layers.add_module('Activation', getattr(nn, activation)())

            in_shape[-1] = features
            shapes.append(in_shape)

        def __getstate__(self) -> dict[str, Any]:
            return super().__getstate__() | {
                'features': self._features,
                'activation': self.layers.Activation.__class__.__name__
                    if 'Activation' in self.layers._modules else None,
            }

Then to allow :class:`~netloader.network.Network` to recognise the custom layer, you can monkey patch the layer into
:ref:`netloader.layers <api_layers>`:

.. code-block:: python

    from netloader import layers

    from src.layers import CustomLinear

    layers.CustomLinear = CustomLinear

Custom Layer From :class:`~netloader.layers.base.BaseMultiLayer`
----------------------------------------------------------------

If your custom layer requires the outputs from multiple layers, then you can inherit from
:class:`~netloader.layers.base.BaseMultiLayer` and implement the ``forward`` method.

.. code-block:: python

    from typing import Any

    from torch import Tensor
    from netloader.utils import Shapes
    from netloader.layers.base import BaseMultiLayer

    class CustomConcat(BaseMultiLayer):
        def __init__(
                self,
                net_check: bool,
                layer: int
                shapes: Shapes,
                check_shapes: Shapes,
                checkpoint: bool = False,
                dim: int = 0,
                **kwargs: Any) -> None:
            super().__init__(
                net_check,
                layer,
                shapes,
                check_shapes,
                checkpoint=checkpoint,
                **kwargs,
            )
            self._dim: int = dim
            out_shape: list[int] = shapes[-1].copy()
            out_shape[self._dim] = out_shape[self._dim] + self._target[self._dim]
            shapes.append(out_shape)

        def __getstate__(self) -> dict[str, Any]:
            return super().__getstate__() | {'dim': self._dim}

        def forward(
                self,
                x: Tensor,
                outputs: list[Tensor],
                checkpoints: list[Tensor],
                *_: Any,
                **_: Any) -> Tensor:
            dim: int

            if self._dim >= 0:
                dim = self._dim + 1
            else:
                dim = self._dim

            if self._checkpoint:
                return torch.cat((x, checkpoints[self._layer]), dim=dim)
            return torch.cat((x, outputs[self._layer]), dim=dim)

        def extra_repr(self) -> str:
            return f'{super().extra_repr()}, dim={self._dim}'

Custom Layer From :class:`~netloader.layers.base.BaseLayer`
-----------------------------------------------------------

If your custom layer requires more complex functionality, then you can inherit from
:class:`~netloader.layers.base.BaseLayer` and implement the ``forward`` method.

.. code-block:: python

    from typing import Any

    from torch import Tensor
    from netloader.utils import Shapes
    from netloader.layers.base import BaseLayer

    class CustomScale(BaseLayer):
        def __init__(
                self,
                scale: float,
                shapes: Shapes,
                **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._scale: Tensor = torch.nn.Parameter(
                torch.tensor(scale),
                requires_grad=True,
            ).to(self._device)
            shapes.append(shapes[-1].copy())

        def __getstate__(self) -> dict[str, Any]:
            return super().__getstate__() | {'scale': self._scale.detach().cpu().tolist()[0]}

        def forward(self, x: Tensor, *_: Any, **_: Any) -> Tensor:
            return x * self._scale

        def extra_repr(self) -> str:
            return f'param={self._scale.detach().cpu().tolist()[0]}'

        def to(self, *args: Any, **kwargs: Any) -> None:
            super().to(*args, **kwargs)
            self._scale = self._scale.to(*args, **kwargs)

Layer Types
===========

All layers currently supported by NetLoader are listed below with a brief description of their functionality and their
initialisation arguments, excluding general initialisation arguments defined in
:class:`~netloader.layers.base.BaseLayer`.
For more details on each layer, please refer to the :ref:`API documentation <api_layers>`.

Blocks
------

Blocks are layers that consist of multiple sub-layers made with Python code.
See :doc:`/api/layers/blocks` for more details.

.. autolayer:: netloader.layers.blocks

Composite
---------

Composites are layers that consist of multiple sub-layers made with ``.json`` files.
See :doc:`/api/layers/composite` for more details.

.. autolayer:: netloader.network.Branch

.. autolayer:: netloader.network.Composite

Convolutional
-------------

Convolutional layers are layers that perform convolution operations.
See :doc:`/api/layers/convolutional` for more details.

.. autolayer:: netloader.layers.convolutional

Flows
-----

Flows are layers that perform normalising flow operations.
See :doc:`/api/layers/flows` for more details.

.. autolayer:: netloader.layers.flows

Linear
------

Linear layers perform simple operations on the input.
See :doc:`/api/layers/linear` for more details.

.. autolayer:: netloader.layers.linear

Miscellaneous
-------------

Miscellaneous layers are layers that do not fit into the other categories.
See :doc:`/api/layers/misc` for more details.

.. autolayer:: netloader.layers.misc

Multi Layers
------------
Layers that perform operations on multiple inputs or outputs.
See :doc:`/api/layers/multi_layer` for more details.

.. autolayer:: netloader.layers.multi_layer

Pooling
-------

Pooling layers perform pooling operations.
See :doc:`/api/layers/pooling` for more details.

.. autolayer:: netloader.layers.pooling

Recurrent
---------

Recurrent layers perform recurrent operations.
See :doc:`/api/layers/recurrent` for more details.

.. autolayer:: netloader.layers.recurrent
