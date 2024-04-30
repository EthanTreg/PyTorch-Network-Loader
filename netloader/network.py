"""
Constructs a network from layers and can load weights to resume network training
"""
import json
import logging as log

import torch
from torch import nn, optim, Tensor

from netloader import layers
from netloader.layers.utils import BaseMultiLayer


class Network(nn.Module):
    """
    Constructs a neural network from a configuration file

    Attributes
    ----------
    name : string
        Name of the network, used for saving
    shapes : list[list[integer]]
        Layer output shapes
    layers : list[dictionary]
        Layers with layer parameters
    checkpoints : list[Tensor]
        Outputs from each checkpoint
    net : ModuleList
        Network construction
    optimiser : Optimizer
        Network optimizer
    scheduler : ReduceLROnPlateau
        Optimizer scheduler
    layer_num : integer, default = None
        Number of layers to use, if None use all layers
    group : integer, default = 0
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
            in_shape: list[int],
            out_shape: list[int],
            learning_rate: float,
            name: str,
            config_dir: str):
        """
        Parameters
        ----------
        in_shape : list[integer]
            Shape of the input tensor, excluding batch size
        out_shape : list[integer]
            shape of the output tensor, excluding batch size
        learning_rate : float
            Optimizer initial learning rate
        name : string
            Name of the network configuration file (without extension)
        config_dir : string
            Path to the network config directory
        """
        super().__init__()
        self.layer_num = None
        self.group = 0
        self.kl_loss_weight = 1e-1
        self.name = name
        self.checkpoints = []
        self.kl_loss = torch.tensor(0.)

        # Construct layers in network
        self._checkpoints, self.shapes, _, self.layers, self.net = _create_network(
            in_shape,
            out_shape,
            f'{config_dir}{name}.json',
        )

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, factor=0.5)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the network

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor with batch size N

        Returns
        -------
        (N,...) Tensor
            Output tensor from the network
        """
        outputs = None
        self.checkpoints = []

        if not self._checkpoints:
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

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.kl_loss = self.kl_loss.to(*args, **kwargs)

        for layer in self.net:
            layer.to(*args, **kwargs)

        return self


class Composite(BaseMultiLayer):
    """
    Creates a subnetwork from a configuration file

    Attributes
    ----------
    layers : ModuleList
        Layers to loop through in the forward pass
    shapes : list[list[integer]]
        Layer output shapes

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the subnetwork
    extra_repr() -> string
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            net_check: bool,
            idx: int,
            name: str,
            config_dir: str,
            shapes: list[list[int]],
            check_shapes: list[list[int]],
            checkpoint: bool = False,
            channels: int = None,
            shape: list[int] = None,
            defaults: dict = None,
            **kwargs):
        """
        Parameters
        ----------
        net_check : boolean
            If layer index should be relative to checkpoint layers
        idx : integer
            Layer number
        name : string
            Name of the subnetwork
        config_dir : string
            Path to the directory with the network configuration file
        shapes : list[integer]
            Shape of the outputs from each layer
        check_shapes : list[list[integer]]
            Shape of the outputs from each checkpoint
        checkpoint : boolean, default = False
            If layer index should be relative to checkpoint layers
        channels : integer, optional
            Number of output channels, won't be used if out_shape is provided, if channels and
            out_shape aren't provided, the input dimensions will be preserved
        shape : list[integer], optional
            Output shape of the block, will be used if provided; otherwise, channels will be used
        defaults : dictionary, default = None
            Default values for the parameters for each type of layer

        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(
            net_check,
            0,
            shapes,
            check_shapes,
            checkpoint=checkpoint,
            idx=idx,
            **kwargs,
        )
        self._name = name
        shapes.append(shapes[-1].copy())

        # Subnetwork output if provided
        if shape:
            shapes[-1] = shape
        elif channels:
            shapes[-1][0] = channels

        if '.json' not in self._name:
            self._name += '.json'

        # Create subnetwork
        _, self.shapes, *_, self.layers = _create_network(
            shapes[-2],
            shapes[-1],
            f'{config_dir}{self._name}',
            suppress_warning=True,
            defaults=defaults,
        )
        shapes[-1] = self.shapes[-1]

    def forward(
            self,
            x: Tensor,
            outputs: list[Tensor],
            checkpoints: list[Tensor],
            net: Network,
            *_) -> Tensor:
        """
        Forward pass of the composite layer

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor with batch size N

        Returns
        -------
        (N,...) Tensor
            Output tensor from the composite layer
        """
        outputs = None
        checkpoints = []

        if not self._checkpoint:
            outputs = [x]

        for i, layer in enumerate(self.layers):
            try:
                x = layer(x, outputs=outputs, checkpoints=checkpoints, net=net)
            except RuntimeError:
                log.error(f'Error in {type(layer).__name__} (layer {i})')
                raise

            if not self._checkpoint:
                outputs.append(x)

        return x

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network

        Returns
        -------
        string
            Layer parameters
        """
        return f'name={self._name}, checkpoint={self._checkpoint}, out_shape={self.shapes[-1]}'


def _create_network(
        in_shape: list[int],
        out_shape: list[int],
        config_path: str,
        suppress_warning: bool = False,
        defaults: dict = None,
) -> tuple[bool, list[list[int]], list[list[int]], list[dict], nn.ModuleList]:
    """
    Creates a network from a config file

    Parameters
    ----------
    in_shape : integer
        Shape of the input tensor, excluding batch size
    out_shape : integer
        Shape of the output tensor, excluding batch size
    config_path : string
        Path to the config file
    suppress_warning : boolean, default = False
        If output shape mismatch warning should be suppressed
    defaults : dictionary, default = None
        Default values for the parameters for each type of layer

    Returns
    -------
    tuple[bool, list[list[integer]], list[dictionary], ModuleList]
        If layer outputs should not be saved, layer output shapes, checkpoint shapes,
        layers in the network with parameters and network construction
    """
    shapes = [in_shape]
    check_shapes = []
    net_check = False
    module_list = nn.ModuleList()

    if defaults is None:
        defaults = {}

    # Load network configuration file
    with open(config_path, 'r', encoding='utf-8') as config:
        config = json.load(config)

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
