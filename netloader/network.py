"""
Constructs a network from layers and can load weights to resume network training
"""
import json

import torch
from torch import nn, optim, Tensor

from netloader import layers


class Network(nn.Module):
    """
    Constructs a neural network from a configuration file

    Attributes
    ----------
    name : string
        Name of the network, used for saving
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
        self.latent_mse_weight = 1e-2
        self.kl_loss_weight = 1e-1
        self.extraction_loss = 1e-1
        self.name = name
        self.clone = None
        self.extraction = None
        self.kl_loss = torch.tensor(0.)

        # Construct layers in network
        self.layers, self.network = _create_network(
            in_shape,
            out_shape,
            f'{config_dir}{name}.json',
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
        outputs = [x]

        for i, layer in enumerate(self.layers):
            # Sampling layer
            if layer['type'] == 'sample':
                x, self.kl_loss = self.network[i](x)
                self.kl_loss *= self.kl_loss_weight
            # Extraction layer
            elif layer['type'] == 'extract':
                self.extraction = x[..., :layer['number']]
                x = x[..., layer['number']:]
            # Cloning layer
            elif layer['type'] == 'clone':
                self.clone = x[..., :layer['number']].clone()
            # Concatenation layers
            elif layer['type'] == 'concatenate':
                if 'dim' in layer and layer['dim'] >= 0:
                    dim = layer['dim'] + 1
                elif 'dim' in layer:
                    dim = layer['dim']
                else:
                    dim = 1

                x = torch.cat((x, outputs[layer['layer']]), dim=dim)
            # Shortcut layers
            elif layer['type'] == 'shortcut':
                x = x + outputs[layer['layer']]
            # Skip layers
            elif layer['type'] == 'skip':
                x = outputs[layer['layer']]
            # All other layers
            else:
                x = self.network[i](x)

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
        dropout_prob : float, optional
            Probability of dropout if dropout from layer is True
    layer : dictionary
        config_path : string
            Path to the .json file containing the block architecture
        channels : integer, optional
            Number of output channels, won't be used if out_shape is provided, if channels and
            out_shape aren't provided, the input dimensions will be preserved
        out_shape : list[integer], optional
            Output shape of the block, will be used if provided; otherwise, channels will be used

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['shape'].append(kwargs['shape'][-1].copy())

    # Subnetwork output
    if 'out_shape' in layer:
        kwargs['shape'][-1] = layer['out_shape']
    elif 'channels' in layer:
        kwargs['shape'][-1][0] = layer['channels']

    # Create subnetwork
    sub_layers, sub_network = _create_network(
        kwargs['shape'][-2],
        kwargs['shape'][-1],
        layer['config_path'],
    )

    # Fix layers that depend on other layers
    for sub_layer in sub_layers:
        if 'layer' in sub_layer and sub_layer['layer'] >= 0:
            sub_layer['layer'] += kwargs['i']

    return kwargs, sub_layers, sub_network


def _create_network(
        in_shape: list[int],
        out_shape: list[int],
        config_path: str) -> tuple[list[dict], nn.ModuleList]:
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

    Returns
    -------
    tuple[list[dictionary], ModuleList]
        Layers in the network with parameters and network construction
    """
    layer_injections = 0

    # Load network configuration file
    with open(config_path, 'r', encoding='utf-8') as file:
        file = json.load(file)

    if '2d' not in file['net']:
        file['net']['2d'] = False

    # Initialize variables
    kwargs = {
        'shape': [in_shape],
        'out_shape': out_shape,
        **file['net'],
    }
    module_list = nn.ModuleList()

    # Create layers
    for i, layer in enumerate(file['layers'].copy()):
        i += layer_injections
        kwargs['i'] = i
        kwargs['module'] = nn.Sequential()

        # If layer is a composite layer, create the subnetwork
        if layer['type'] == 'composite':
            kwargs, sub_layers, sub_network = _composite_layer(kwargs, layer)
            module_list.extend(sub_network)

            # Replace composite layer with the subnetwork
            del file['layers'][i]
            file['layers'][i:i] = sub_layers
            layer_injections += len(sub_layers) - 1
        else:
            kwargs = getattr(layers, layer['type'])(kwargs, layer)
            module_list.append(kwargs['module'])

    # Check network output dimensions
    if kwargs['shape'][-1] != out_shape:
        raise ValueError(
            f"Network output shape {kwargs['shape'][-1]} != data output shape {out_shape}"
        )

    return file['layers'], module_list
