"""
Constructs a network from layers and can load weights to resume network training
"""
import os
import json
import logging as log
from warnings import warn
from typing import Any, TextIO, Self

import torch
import numpy as np
from torch import nn, Tensor

from netloader import layers
from netloader.layers.base import BaseLayer
from netloader.utils.utils import check_params, deep_merge


class Network(nn.Module):
    """
    Constructs a neural network from a configuration file

    Attributes
    ----------
    name : str
        Name of the network, used for saving
    check_shapes : list[list[int]]
        Checkpoint output shapes
    shapes : list[list[int] | list[list[int]]
        Layer output shapes
    checkpoints : list[Tensor]
        Outputs from each checkpoint
    config : dict[str, Any]
        Network configuration
    net : ModuleList
        Network construction
    layer_num : int, default = None
        Number of layers to use, if None use all layers
    group : int, default = 0
        Which group is the active group if a layer has the group attribute
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
            config: str | dict[str, Any],
            in_shape: list[int] | list[list[int]],
            out_shape: list[int],
            suppress_warning: bool = False,
            defaults: dict[str, Any] | None = None) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the network configuration file
        config : str | dict[str, Any]
            Path to the network config directory or configuration dictionary
        in_shape : list[int] | list[list[int]],
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        suppress_warning : bool, default = False
            If output shape mismatch warning should be suppressed
        defaults : dict[str, Any], default = None
            Default values for the parameters for each type of layer
        """
        super().__init__()
        self._checkpoints: bool
        self.group: int = 0
        self.layer_num: int | None = None
        self.name: str = name
        self.check_shapes: list[list[int]]
        self.shapes: list[list[int] | list[list[int]]]
        self.config: dict[str, Any]
        self.checkpoints: list[Tensor] = []
        self.kl_loss: Tensor = torch.tensor(0.)
        self.net: nn.ModuleList

        if '.json' not in self.name:
            self.name += '.json'

        # Construct layers in network
        self._checkpoints, self.shapes, self.check_shapes, self.config, self.net = _create_network(
            os.path.join(config, self.name) if isinstance(config, str) else config,
            in_shape,
            out_shape,
            suppress_warning=suppress_warning,
            defaults=defaults,
        )

        self.name = self.name.replace('.json', '')

        # Adds Network class to list of safe PyTorch classes when loading saved networks
        torch.serialization.add_safe_globals([self.__class__])

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the network for pickling

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the network
        """
        return {
            'checkpoints': self._checkpoints,
            'group': self.group,
            'layer_num': self.layer_num,
            'name': self.name,
            'check_shapes': self.check_shapes,
            'shapes': self.shapes,
            'config': self.config,
            'net': self.cpu().state_dict(),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the network for pickling

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the network
        """
        super().__init__()
        self.group = state['group']
        self.layer_num = state['layer_num']
        self.name = state['name']
        self.kl_loss = torch.tensor(0.)

        self._checkpoints, self.shapes, self.check_shapes, self.config, self.net = _create_network(
            state['config'],
            state['shapes'][0],
            state['shapes'][-1],
            suppress_warning=True,
        )

        try:
            self.load_state_dict(state['net'])
        except RuntimeError:
            if 'net' not in list(state['net'].keys())[0]:
                warn(
                    'Network state is saved in old non-weights safe format and is '
                    'deprecated, please resave the network in the new format using net.save()',
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                log.getLogger().warning('Failed to load network, incorrect layer names, will try '
                                        'to match weights.')

            self.load_state_dict(dict(zip(self.state_dict(), state['net'].values())))

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

        for i, layer in enumerate(self.config['layers'][:self.layer_num]):
            if 'group' in layer and layer['group'] != self.group:
                outputs.append(torch.tensor([]))

                if layer['type'] == 'Checkpoint':
                    self.checkpoints.append(torch.tensor([]))
                continue

            try:
                x = self.net[i](x, outputs=outputs, checkpoints=self.checkpoints, net=self)
            except RuntimeError:
                log.getLogger(__name__).error(f"Error in {layer['type']} (layer {i})")
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
            shapes: list[list[int] | list[list[int]]],
            checkpoint: bool = True,
            channels: int | None = None,
            config_dir: str = '',
            shape: list[int] | None = None,
            config: dict[str, Any] | None = None,
            defaults: dict[str, Any] | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        name : str
            Name of the subnetwork
        shapes : list[list[int] | list[list[int]]]
            Shape of the outputs from each layer
        checkpoint : bool, default = True
            If layer index should be relative to checkpoint layers
        channels : int, optional
            Number of output channels, won't be used if out_shape is provided, if channels and
            out_shape aren't provided, the input dimensions will be preserved
        config_dir : str, default = ''
            Path to the directory with the network configuration file, won't be used if config is
            provided
        shape : list[int], optional
            Output shape of the block, will be used if provided; otherwise, channels will be used
        config : dict[str, Any] | None, default = None
            Network configuration dictionary
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
            config or config_dir,
            shapes[-2],
            shapes[-1],
            suppress_warning=True,
            defaults=defaults | {'checkpoints': self._checkpoint},
        )
        shapes[-1] = self.net.shapes[-1]

    def forward(self, x: list[Tensor] | Tensor, *_: Any, **__: Any) -> Tensor:
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
        config: str | dict[str, Any],
        in_shape: list[int] | list[list[int]],
        out_shape: list[int],
        suppress_warning: bool = False,
        defaults: dict[str, Any] | None = None,
) -> tuple[bool, list[list[int] | list[list[int]]], list[list[int]], dict[str, Any], nn.ModuleList]:
    """
    Creates a network from a config file

    Parameters
    ----------
    config : str | dict[str, Any]
        Path to the config file or configuration dictionary
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
    net_check: bool = False
    config_path: str = config if isinstance(config, str) else 'config'
    check_shapes: list[list[int]] = []
    shapes: list[list[int] | list[list[int]]] = [in_shape]
    file: TextIO
    logger: log.Logger = log.getLogger(__name__)
    module_list: nn.ModuleList = nn.ModuleList()

    defaults = defaults or {}

    # Load network configuration file
    if isinstance(config, str):
        with open(config, 'r', encoding='utf-8') as file:
            config = json.load(file)

    assert isinstance(config, dict)

    if 'checkpoints' in config['net']:
        net_check = config['net']['checkpoints']

    config['net'] = deep_merge(config['net'], defaults)

    # Check for unknown net parameters
    if not suppress_warning:
        check_params(
            f"'net' in {config_path.split('/')[-1]}",
            ['checkpoints', 'Composite', 'description', 'paper', 'github'] + dir(layers),
            np.array(list(config['net'].keys())),
        )

    # Create layers
    for i, layer in enumerate(config['layers'].copy()):
        if layer['type'] in config['net']:
            layer = config['net'][layer['type']] | layer

        if layer['type'] == 'Composite':
            layer_class = Composite
        else:
            layer_class = getattr(layers, layer['type'])

        try:
            module_list.append(layer_class(
                net_check=net_check,
                idx=i,
                net_out=out_shape,
                shapes=shapes,
                check_shapes=check_shapes,
                **layer,
            ))
        except ValueError:
            logger.error(f"Error in {layer['type']} (layer {i})")
            raise

        module_list[-1].initialise_layers()

        if layer_class == Composite:
            config['layers'][i]['config'] = module_list[-1].net.config

    # Check network output dimensions
    if not suppress_warning and shapes[-1] != out_shape:
        logger.warning(
            f'Network output shape {shapes[-1]} != data output shape {out_shape}'
        )

    return net_check, shapes, check_shapes, config, module_list
