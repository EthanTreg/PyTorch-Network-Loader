"""
Constructs a network from layers and can load weights to resume network training
"""
import json
import logging as log

import torch
from torch import nn, optim, Tensor

from netloader.utils import layers
from netloader.utils.utils import get_device


class Network(nn.Module):
    """
    Constructs a neural network from a configuration file

    Attributes
    ----------
    layers : list[dictionary]
        Layers with layer parameters
    clone : Tensor
        Cloned values from the network
    extraction : Tensor
        Values extracted from the network using an extraction layer
    network : ModuleList
        Network construction
    optimizer : Optimizer
        Network optimizer
    scheduler : ReduceLROnPlateau
        Optimizer scheduler

    Methods
    -------
    forward(x)
        Forward pass of the network
    """
    def __init__(
            self,
            in_size: int,
            out_size: int,
            learning_rate: float,
            name: str,
            config_dir: str):
        """
        Parameters
        ----------
        in_size : integer
            Size of the input tensor
        out_size : integer
            Size of the output tensor
        learning_rate : float
            Optimizer initial learning rate
        name : string
            Name of the network, used for saving
        config_dir : string
            Path to the network config directory
        """
        super().__init__()
        self.latent_mse_weight = 1e-2
        self.kl_loss_weight = 1e-1
        self.extraction_loss = 1e-1
        self.clone = None
        self.extraction = None
        self.kl_loss = torch.tensor(0.)

        # Construct layers in CNN
        self.layers, self.network = create_network(
            in_size,
            out_size,
            f'{config_dir}{name}.json',
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
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
    network.optimizer.load_state_dict(state['optimizer'])
    network.scheduler.load_state_dict(state['scheduler'])
    train_loss = state['train_loss']
    val_loss = state['val_loss']

    return initial_epoch, network, (train_loss, val_loss)


def create_network(
        in_size: int | tuple[int, int],
        out_size: int | tuple[int, int],
        config_path: str) -> tuple[list[dict], nn.ModuleList]:
    """
    Creates a network from a config file

    Parameters
    ----------
    in_size : integer
        Size of the input
    out_size : integer
        Size of the spectra
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

    if isinstance(in_size, tuple) and len(in_size) == 2:
        dims, in_size = in_size
    else:
        dims = in_size

    if isinstance(out_size, tuple):
        out_size = out_size[-1]

    # Initialize variables
    kwargs = {
        'data_size': [in_size],
        'output_size': out_size,
        'dims': [dims],
        **file['net'],
    }
    module_list = nn.ModuleList()

    # Create layers
    for i, layer in enumerate(file['layers']):
        kwargs['i'] = i
        kwargs['module'] = nn.Sequential()

        try:
            kwargs = getattr(layers, layer['type'])(kwargs, layer)
        except AttributeError as error:
            log.error(f"Unknown layer: {layer['type']}")
            raise error

        module_list.append(kwargs['module'])

    if kwargs['data_size'][-1] != out_size:
        log.error(
            f"Network output size ({kwargs['data_size']}) != data output size ({out_size})",
        )
    elif kwargs['dims'][-1] != out_size:
        log.error(f"Network output filters (num={kwargs['dims'][-1]}) has not been reduced, "
                  'reshape with output = [-1] may be missing')

    return file['layers'], module_list
