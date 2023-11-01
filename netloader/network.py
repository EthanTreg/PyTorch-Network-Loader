"""
Constructs a network from layers and can load weights to resume network training
"""
import json

import torch
from torch import nn, optim, Tensor

from netloader import layers
from netloader.utils.utils import get_device


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
            Shape of the input tensor
        out_shape : list[integer]
            shape of the output tensor
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
                if 'channel' in layer and layer['channel']:
                    dim = 1
                else:
                    dim = -1

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


def _create_network(
        in_shape: list[int],
        out_shape: list[int],
        config_path: str) -> tuple[list[dict], nn.ModuleList]:
    """
    Creates a network from a config file

    Parameters
    ----------
    in_shape : integer
        Shape of the input tensor
    out_shape : integer
        Shape of the output tensor
    config_path : string
        Path to the config file

    Returns
    -------
    tuple[list[dictionary], ModuleList]
        Layers in the network with parameters and network construction
    """
    # Load network configuration file
    with open(config_path, 'r', encoding='utf-8') as file:
        file = json.load(file)

    if '2d' not in file['net']:
        file['net']['2d'] = False

    # Initialize variables
    kwargs = {
        'shape': [in_shape],
        'output_shape': out_shape,
        **file['net'],
    }
    module_list = nn.ModuleList()

    # Create layers
    for i, layer in enumerate(file['layers']):
        kwargs['i'] = i
        kwargs['module'] = nn.Sequential()
        kwargs = getattr(layers, layer['type'])(kwargs, layer)
        module_list.append(kwargs['module'])

    # Check network output dimensions
    if kwargs['shape'][-1] != out_shape:
        raise ValueError(
            f"Network output shape {kwargs['shape'][-1]} != data output shape {out_shape}"
        )

    return file['layers'], module_list


def load_network(
        load_num: int,
        states_dir: str,
        network: Network) -> tuple[int, Network, tuple[list, list]] | None:
    """
    Loads the network from a previously saved state

    Can account for changes in the network

    Parameters
    ----------
    load_num : integer
        File number of the saved state
    states_dir : string
        Directory to the save files
    network : Network
        The network to append saved state to

    Returns
    -------
    tuple[int, Encoder | Decoder, Optimizer, ReduceLROnPlateau, tuple[list, list]]
        The initial epoch, the updated network, optimizer
        and scheduler, and the training and validation losses
    """
    state = torch.load(f'{states_dir}{network.name}_{load_num}.pth', map_location=get_device()[1])

    # Apply the saved states to the new network
    initial_epoch = state['epoch']
    network.load_state_dict(network.state_dict() | state['state_dict'])
    network.optimiser.load_state_dict(state['optimizer'])
    network.scheduler.load_state_dict(state['scheduler'])
    train_loss = state['train_loss']
    val_loss = state['val_loss']

    return initial_epoch, network, (train_loss, val_loss)
