"""
Linear network layers
"""
import torch
from torch import nn, Tensor

from netloader.utils.utils import get_device
from netloader.layers.utils import optional_layer


class Sample(nn.Module):
    """
    Samples random values from a Gaussian distribution for a variational autoencoder

    Attributes
    ----------
    mean_layer : Module
        Linear layer to predict values for the mean of the distribution
    std_layer : Module
        Linear layer to predict values for the standard deviation of the distribution
    sample_layer : Module
        Layer to sample values from a Gaussian distribution

    Methods
    -------
    forward(x)
        Forward pass of the sampling layer
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : integer
            Number of input features for linear layer
        out_features : integer
            Number of output features for linear layer
        """
        super().__init__()
        device = get_device()[1]

        self.mean_layer = nn.Linear(in_features=in_features, out_features=out_features)
        self.std_layer = nn.Linear(in_features=in_features, out_features=out_features)
        self.sample_layer = torch.distributions.Normal(
            torch.tensor(0.).to(device),
            torch.tensor(1.).to(device),
        ).sample

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the sampling layer for a variational autoencoder

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        tuple[Tensor, Tensor]
            Output tensor and KL Divergence loss
        """
        mean = self.mean_layer(x)
        std = torch.exp(self.std_layer(x))
        x = mean + std * self.sample_layer(mean.shape)
        kl_loss = 0.5 * torch.mean(mean ** 2 + std ** 2 - 2 * torch.log(std) - 1)

        return x, kl_loss


def linear(kwargs: dict, layer: dict) -> dict:
    """
    Linear layer constructor

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
        output_size : integer, optional
            Size of the network's output, required only if layer contains factor and not features;
        dropout_prob : float, optional
            Probability of dropout if dropout from layer is True;
    layer : dictionary
        factor : float, optional
            Output features is equal to the factor of the network's output,
            will be used if provided, else features will be used;
        features : integer, optional
            Number of output features for the layer,
            if output_size from kwargs and factor is provided, features will not be used;
        dropout : boolean, default = False
            If dropout should be used;
        batch_norm : boolean, default = 0
            If batch normalisation should be used;
        activation : boolean, default = True
            If SELU activation should be used;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    # Number of features can be defined by either a factor of the output size or explicitly
    try:
        kwargs['dims'].append(int(kwargs['output_size'] * layer['factor']))
    except KeyError:
        kwargs['dims'].append(layer['features'])

    linear_layer = nn.Linear(in_features=kwargs['dims'][-2], out_features=kwargs['dims'][-1])
    kwargs['module'].add_module(f"linear_{kwargs['i']}", linear_layer)

    # Optional layers
    optional_layer(False, 'dropout', kwargs, layer, nn.Dropout1d(kwargs['dropout_prob']))
    optional_layer(False, 'batch_norm', kwargs, layer, nn.BatchNorm1d(kwargs['dims'][-1]))
    optional_layer(True, 'activation', kwargs, layer, nn.SELU())

    # Data size equals number of nodes
    kwargs['data_size'].append(kwargs['dims'][-1])

    return kwargs


def upsample(kwargs: dict, _: dict) -> dict:
    """
    Constructs a 2x upscaler using linear upsampling

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
    _ : dictionary
        For compatibility

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    kwargs['dims'].append(kwargs['dims'][-1])
    kwargs['module'].add_module(
        f"upsample_{kwargs['i']}",
        nn.Upsample(scale_factor=2, mode='linear')
    )
    # Data size doubles
    kwargs['data_size'].append(kwargs['data_size'][-1] * 2)

    return kwargs


def sample(kwargs: dict, layer: dict) -> dict:
    """
    Generates mean and standard deviation and randomly samples from a Gaussian distribution
    for a variational autoencoder

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number;
        data_size : list[integer]
            Hidden layer output length;
        dims : list[integer]
            Dimensions in each layer, either linear output features or convolutional/GRU filters;
        module : Sequential
            Sequential module to contain the layer;
        output_size : integer, optional
            Size of the network's output, required only if layer contains factor and not features;
    layer : dictionary
        factor : float, optional
            Output features is equal to the factor of the network's output,
            will be used if provided, else features will be used;
        features : integer, optional
            Number of output features for the layer,
            if output_size from kwargs and factor is provided, features will not be used;

    Returns
    -------
    dictionary
        Returns the input kwargs with any changes made by the function
    """
    # Number of features can be defined by either a factor of the output size or explicitly
    try:
        kwargs['dims'].append(int(kwargs['output_size'] * layer['factor']))
    except KeyError:
        kwargs['dims'].append(layer['features'])

    kwargs['data_size'].append(kwargs['dims'][-1])
    kwargs['module'].add_module(
        f"sample_{kwargs['i']}",
        Sample(kwargs['dims'][-2], kwargs['dims'][-1]),
    )
    return kwargs