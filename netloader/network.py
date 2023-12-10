"""
Constructs a network from layers and can load weights to resume network training
"""
import json
import logging as log

import torch
from torch import nn, optim, Tensor

from netloader import layers
from netloader.utils.defaults import DEFAULTS


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
    kl_loss : Tensor
        KL divergence loss on the latent space, if using a sample layer
    clone : Tensor
        Cloned values from the network if using a sample layer
    extraction : Tensor
        Values extracted from the network1 if using an extraction layer
    network : ModuleList
        Network construction
    optimiser : Optimizer
        Network optimizer
    scheduler : ReduceLROnPlateau
        Optimizer scheduler
    checkpoints : boolean, default = False
        If outputs from each layer should not be saved to reduce memory usage, but with more limited
        layer output referencing
    layer_num : integer, default = None
        Number of layers to use, if None use all layers
    latent_mse_weight : float, default = 1e-2
        Relative weight if performing an MSE loss on the latent space
    kl_loss_weight : float, default = 1e-1
        Relative weight if performing a KL divergence loss on the latent space
    extraction_loss : float, default = 1e-1
        Relative weight if performing a loss on the extracted features

    Methods
    -------
    forward(x)
        Forward pass of the network
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
        self.latent_mse_weight = 1e-2
        self.kl_loss_weight = 1e-1
        self.extraction_loss = 1e-1
        self.name = name
        self.clone = None
        self.extraction = None
        self.kl_loss = torch.tensor(0.)

        # Construct layers in network
        self._checkpoints, self.shapes, _, self.layers, self.network = _create_network(
            in_shape,
            out_shape,
            f'{config_dir}{name}.json',
            defaults=DEFAULTS,
        )

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            factor=0.5,
            verbose=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the network

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Output tensor from the network
        """
        checkpoints = []

        if not self._checkpoints:
            outputs = [x]

        for i, layer in enumerate(self.layers[:self.layer_num]):
            # Checkpoint layer
            if layer['type'] == 'checkpoint':
                checkpoints.append(x)
            # Cloning layer
            elif layer['type'] == 'clone':
                self.clone = x[..., :layer['number']].clone()
            # Concatenation layers
            elif layer['type'] == 'concatenate':
                if ('checkpoint' in layer and layer['checkpoint']) or self._checkpoints:
                    x = _concatenate(layer, x, checkpoints)
                else:
                    x = _concatenate(layer, x, outputs)
            # Extraction layer
            elif layer['type'] == 'extract':
                self.extraction = x[..., :layer['number']]
                x = x[..., layer['number']:]
            # Sampling layer
            elif layer['type'] == 'sample':
                x, self.kl_loss = self.network[i](x)
                self.kl_loss *= self.kl_loss_weight
            # Shortcut layers
            elif layer['type'] == 'shortcut':
                if ('checkpoint' in layer and layer['checkpoint']) or self._checkpoints:
                    x = x + checkpoints[layer['layer']]
                else:
                    x = x + outputs[layer['layer']]
            # Skip layers
            elif layer['type'] == 'skip':
                if ('checkpoint' in layer and layer['checkpoint']) or self._checkpoints:
                    x = checkpoints[layer['layer']]
                else:
                    x = outputs[layer['layer']]
            # All other layers
            else:
                x = self.network[i](x)

            if not self._checkpoints:
                outputs.append(x)

        return x


def _composite_layer(kwargs: dict, layer: dict) -> tuple[dict, list[dict], nn.ModuleList]:
    """
    Creates a subnetwork from a configuration file

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
        module : Sequential
            Sequential module to contain the layer
        out_shape : list[integer], optional
            Shape of the network's output, required only if layer contains factor
    layer : dictionary
        config_path : string
            Path to the .json file containing the block architecture
        channels : integer, optional
            Number of output channels, won't be used if out_shape is provided, if channels and
            out_shape aren't provided, the input dimensions will be preserved
        out_shape : list[integer], optional
            Output shape of the block, will be used if provided; otherwise, channels will be used
        defaults : dictionary, optional
            Default values for each layer that override the network's
            default values, the dictionary contains sub-dictionaries named layer_name,
            which contain the parameters found in the layers above with the corresponding layer name

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    suppress_warning = False
    defaults = DEFAULTS.copy()
    kwargs['shape'].append(kwargs['shape'][-1].copy())

    # Subnetwork output if provided
    if 'out_shape' in layer:
        kwargs['shape'][-1] = layer['out_shape']
    elif 'channels' in layer:
        kwargs['shape'][-1][0] = layer['channels']
    else:
        suppress_warning = True

    # Override default values if composite layer has default values
    if 'defaults' in layer.keys():
        for key in layer['defaults'].keys():
            defaults[key] = defaults[key] | layer['defaults'][key]

    # Create subnetwork
    checkpoints, shapes, check_shapes, sub_layers, sub_network = _create_network(
        kwargs['shape'][-2],
        kwargs['shape'][-1],
        layer['config_path'],
        suppress_warning=suppress_warning,
        defaults=defaults,
    )

    # Fix layers that depend on other layers
    for sub_layer in sub_layers:
        if 'layer' not in sub_layer or (sub_layer['layer'] < 0 and not checkpoints):
            continue

        if sub_layer['layer'] < 0:
            sub_layer['checkpoint'] = 1
        elif checkpoints or ('checkpoint' in layer and layer['checkpoint']):
            sub_layer['layer'] += len(kwargs['check_shape'])
            sub_layer['checkpoint'] = 1
        else:
            sub_layer['layer'] += kwargs['i']

    kwargs['shape'][-1] = shapes[-1]
    kwargs['check_shape'].extend(check_shapes)
    return kwargs, sub_layers, sub_network


def _concatenate(layer: dict, x: Tensor, outputs: list[Tensor]) -> Tensor:
    """
    Concatenates the output from the previous layer and from the specified layer along a
    specified dimension

    Parameters
    ----------
    layer : dictionary
        layer : integer
            Layer index to concatenate the previous layer output with
        dim : integer, default = 0
            Dimension to concatenate to
    x : Tensor
        Output from the previous layer
    outputs : list[Tensor]
        Layer outputs to index for concatenation

    Returns
    -------
    Tensor
        Concatenated tensor
    """
    if 'dim' in layer and layer['dim'] >= 0:
        dim = layer['dim'] + 1
    elif 'dim' in layer:
        dim = layer['dim']
    else:
        dim = 1

    return torch.cat((x, outputs[layer['layer']]), dim=dim)


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

    Returns
    -------
    tuple[bool, list[list[integer]], list[dictionary], ModuleList]
        If layer outputs should not be saved, layer output shapes, checkpoint shapes,
        layers in the network with parameters and network construction
    """
    layer_injections = 0

    # Load network configuration file
    with open(config_path, 'r', encoding='utf-8') as config:
        config = json.load(config)

    if '2d' not in config['net']:
        config['net']['2d'] = False

    # Override defaults from net parameters
    for key, value in config['net'].items():
        if isinstance(value, dict):
            defaults[key] = defaults[key] | value
        else:
            defaults[key] = value

    kwargs = defaults | {
        'shape': [in_shape],
        'check_shape': [],
        'out_shape': out_shape,
    }
    module_list = nn.ModuleList()

    # Create layers
    for i, layer in enumerate(config['layers'].copy()):
        i += layer_injections
        kwargs['i'] = i
        kwargs['module'] = nn.Sequential()

        # If layer is a composite layer, create the subnetwork
        if layer['type'] == 'composite':
            kwargs, sub_layers, sub_network = _composite_layer(kwargs, layer)
            module_list.extend(sub_network)

            # Replace composite layer with the subnetwork
            del config['layers'][i]
            config['layers'][i:i] = sub_layers
            layer_injections += len(sub_layers) - 1
        else:
            getattr(layers, layer['type'])(kwargs, layer)
            module_list.append(kwargs['module'])

    # Check network output dimensions
    if not suppress_warning and kwargs['shape'][-1] != out_shape:
        log.warning(
            f"Network output shape {kwargs['shape'][-1]} != data output shape {out_shape}"
        )

    return (
        kwargs['checkpoints'],
        kwargs['shape'],
        kwargs['check_shape'],
        config['layers'],
        module_list,
    )
