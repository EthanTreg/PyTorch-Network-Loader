"""
Constructs a network from layers and can load weights to resume network training
"""
import json
import logging as log
from typing import Any, TextIO, Self

import torch
from torch import nn, optim, Tensor

from netloader import layers
from netloader.layers.utils import BaseLayer


class Network(nn.Module):
    """
    Constructs a neural network from a configuration file

    Attributes
    ----------
    name : str
        Name of the network, used for saving
    shapes : list[list[int] | list[list[int]]
        Layer output shapes
    check_shapes : list[list[int]]
        Checkpoint output shapes
    layers : list[dict[str, Any]]
        Layers with layer parameters
    checkpoints : list[Tensor]
        Outputs from each checkpoint
    net : ModuleList
        Network construction
    optimiser : Optimiser
        Network optimiser, uses Adam optimiser
    scheduler : LRScheduler
        Optimiser scheduler, uses reduce learning rate on plateau
    layer_num : int, default = None
        Number of layers to use, if None use all layers
    group : int, default = 0
        Which group is the active group if a layer has the group attribute
    kl_loss_weight : float, default = 1e-1
        Relative weight if performing a KL divergence loss on the latent space
    kl_loss : Tensor, default = 0
        KL divergence loss on the latent space, if using a sample layer

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the network
    to(*args, **kwargs) -> Network
        Moves and/or casts the parameters and buffers
    """
    def __init__(
            self,
            name: str,
            config_dir: str,
            in_shape: list[int] | list[list[int]],
            out_shape: list[int],
            suppress_warning: bool = False,
            learning_rate: float = 1e-2,
            defaults: dict[str, Any] | None = None):
        """
        Parameters
        ----------
        name : str
            Name of the network configuration file (without extension)
        config_dir : str
            Path to the network config directory
        in_shape : list[int] | list[list[int]],
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        suppress_warning : bool, default = False
            If output shape mismatch warning should be suppressed
        learning_rate : float, default = 1e-2
            Optimiser initial learning rate, if None, no optimiser or scheduler will be set
        defaults : dict[str, Any], default = None
            Default values for the parameters for each type of layer
        """
        super().__init__()
        self._checkpoints: bool
        self.group: int = 0
        self.layer_num: int | None = None
        self.kl_loss_weight: float = 1e-1
        self.name: str = name
        self.check_shapes: list[list[int]]
        self.shapes: list[list[int] | list[list[int]]]
        self.layers: list[dict[str, Any]]
        self.checkpoints: list[Tensor] = []
        self.kl_loss: Tensor = torch.tensor(0.)
        self.net: nn.ModuleList
        self.optimiser: optim.Optimizer
        self.scheduler: optim.lr_scheduler.LRScheduler

        if '.json' not in self.name:
            self.name += '.json'

        # Construct layers in network
        self._checkpoints, self.shapes, self.check_shapes, self.layers, self.net = _create_network(
            f'{config_dir}{self.name}',
            in_shape,
            out_shape,
            suppress_warning=suppress_warning,
            defaults=defaults,
        )

        self.name = self.name.replace('.json', '')

        if learning_rate:
            self.optimiser = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, factor=0.5)

    def forward(self, x: list[Tensor] | Tensor) -> list[Tensor] | Tensor:
        """
        Forward pass of the network

        Parameters
        ----------
        x : (N,...) list[Tensor] | Tensor
            Input tensor(s) with batch size N

        Returns
        -------
        (N,...) list[Tensor] | Tensor
            Output tensor from the network
        """
        self.checkpoints = []
        outputs: list[list[Tensor] | Tensor] = []

        if not self._checkpoints or any(isinstance(layer, layers.Unpack) for layer in self.net):
            outputs = [x]

        for i, layer in enumerate(self.layers[:self.layer_num]):
            if 'group' in layer and layer['group'] != self.group:
                continue

            try:
                x = self.net[i](x, outputs=outputs, checkpoints=self.checkpoints, net=self)
            except RuntimeError:
                log.error(f"Error in {layer['type']} (layer {i})")
                raise

            if not self._checkpoints:
                outputs.append(x)
        return x

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self.kl_loss = self.kl_loss.to(*args, **kwargs)

        for layer in self.net:
            layer.to(*args, **kwargs)

        return self


class Composite(BaseLayer):
    """
    Creates a subnetwork from a configuration file

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers part of the parent layer to loop through in the forward pass
    net : Network
        Subnetwork for the Composite layer

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the subnetwork
    to(*args, **kwargs) -> Composite
        Moves and/or casts the parameters and buffers
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            net_check: bool,
            name: str,
            config_dir: str,
            shapes: list[list[int] | list[list[int]]],
            checkpoint: bool = True,
            channels: int | None = None,
            shape: list[int] | None = None,
            defaults: dict[str, Any] | None = None,
            **kwargs: Any):
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        name : str
            Name of the subnetwork
        config_dir : str
            Path to the directory with the network configuration file
        shapes : list[list[int] | list[list[int]]]
            Shape of the outputs from each layer
        checkpoint : bool, default = True
            If layer index should be relative to checkpoint layers
        channels : int, optional
            Number of output channels, won't be used if out_shape is provided, if channels and
            out_shape aren't provided, the input dimensions will be preserved
        shape : list[int], optional
            Output shape of the block, will be used if provided; otherwise, channels will be used
        defaults : dict[str, Any], default = None
            Default values for the parameters for each type of layer

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._checkpoint: bool
        self.net: Network

        self._checkpoint = checkpoint or net_check
        shapes.append(shapes[-1].copy())

        if defaults is None:
            defaults = {}

        # Subnetwork output if provided
        if shape:
            shapes[-1] = shape
        elif channels:
            assert isinstance(shapes[-1][0], int)
            shapes[-1][0] = channels

        # Create subnetwork
        self.net = Network(
            name,
            config_dir,
            shapes[-2],
            shapes[-1],
            suppress_warning=True,
            defaults=defaults | {'checkpoints': self._checkpoint},
        )
        shapes[-1] = self.net.shapes[-1]

    def forward(self, x: list[Tensor] | Tensor, **_: Any) -> Tensor:
        """
        Forward pass for the Composite layer

        Parameters
        ----------
        x : (N,...) list[Tensor] | Tensor
            Input tensor(s) with batch size N

        Returns
        -------
        (N,...) Tensor
            Output tensor from the subnetwork
        """
        return self.net(x)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        self.net.to(*args, **kwargs)
        return self

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        str
            Layer parameters
        """
        return (f'name={self.net.name}, checkpoint={self._checkpoint}, '
                f'out_shape={self.net.shapes[-1]}')


def _create_network(
        config_path: str,
        in_shape: list[int] | list[list[int]],
        out_shape: list[int],
        suppress_warning: bool = False,
        defaults: dict[str, Any] | None = None,
) -> tuple[bool, list[list[int] | list[list[int]]], list[list[int]], list[dict], nn.ModuleList]:
    """
    Creates a network from a config file

    Parameters
    ----------
    config_path : str
        Path to the config file
    in_shape : list[int] | list[list[int]]
        Shape of the input tensor(s), excluding batch size
    out_shape : list[int]
        Shape of the output tensor, excluding batch size
    suppress_warning : bool, default = False
        If output shape mismatch warning should be suppressed
    defaults : dict[str, Any], default = None
        Default values for the parameters for each type of layer

    Returns
    -------
    tuple[bool, list[list[int] | list[list[int]]], list[dict], ModuleList]
        If layer outputs should not be saved, layer output shapes, checkpoint shapes,
        layers in the network with parameters and network construction
    """
    net_check: bool
    shapes: list[list[int] | list[list[int]]]
    check_shapes: list[list[int]]
    config: dict[str, Any]
    file: TextIO
    module_list: nn.ModuleList

    shapes = [in_shape]
    check_shapes = []
    net_check = False
    module_list = nn.ModuleList()

    if defaults is None:
        defaults = {}

    # Load network configuration file
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)

    if 'checkpoints' in config['net']:
        net_check = config['net']['checkpoints']

    config['net'] = config['net'] | defaults

    # Create layers
    for i, layer in enumerate(config['layers'].copy()):
        if layer['type'] in config['net']:
            layer = config['net'][layer['type']] | layer

        if layer['type'] == 'Composite':
            layer_class = Composite
        else:
            layer_class = getattr(layers, layer['type'])

        module_list.append(layer_class(
            net_check=net_check,
            idx=i,
            net_out=out_shape,
            shapes=shapes,
            check_shapes=check_shapes,
            **layer,
        ))
        module_list[-1].initialise_layers()

    # Check network output dimensions
    if not suppress_warning and shapes[-1] != out_shape:
        log.warning(
            f'Network output shape {shapes[-1]} != data output shape {out_shape}'
        )

    return net_check, shapes, check_shapes, config['layers'], module_list
