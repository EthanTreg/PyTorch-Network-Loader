"""
Linear network layers
"""
import torch
import numpy as np
from torch import nn, Tensor

from netloader.layers.misc import Reshape
from netloader.utils.utils import get_device
from netloader.layers.utils import optional_layer, check_layer


class OrderedBottleneck(nn.Module):
    """
    Information-ordered bottleneck to randomly change the size of the bottleneck in an autoencoder
    to encode the most important information in the first values of the latent space

    Attributes
    ----------
    device : Device
        Which device to use

    Methods
    -------
    forward(x)
        Forward pass of the information-ordered bottleneck layer
    """
    def __init__(self):
        super().__init__()
        self.device = get_device()[1]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the information-ordered bottleneck layer

        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Input zeroed from a random index to the last value
        """
        idx = np.random.randint(x.size(-1))
        gate = torch.zeros(x.size(-1)).to(self.device)
        gate[:idx + 1] = 1
        return x * gate


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


def ordered_bottleneck(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Information-ordered bottleneck to randomly change the size of the bottleneck in an autoencoder
    to encode the most important information in the first values of the latent space

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
        module : Sequential
            Sequential module to contain the layer
    layer : dictionary
        For compatibility
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    check_layer([], kwargs, layer, check_params=check_params)
    kwargs['shape'].append(kwargs['shape'][-1].copy())
    kwargs['module'].add_module(f"ordered_bottleneck_{kwargs['i']}", OrderedBottleneck())


def linear(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Linear layer constructor

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
        features : integer, optional
            Number of output features for the layer,
            if out_shape from kwargs and factor is provided, features will not be used
        factor : float, optional
            Output features is equal to the factor of the network's output,
            will be used if provided, else features will be used
        dropout : float, default =  0.01
            Probability of dropout
        batch_norm : boolean, default = 0
            If batch normalisation should be used
        activation : boolean, default = True
            If SELU activation should be used
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    supported_params = ['features', 'factor', 'dropout', 'batch_norm', 'activation']
    layer = check_layer(supported_params, kwargs, layer, check_params=check_params)

    in_shape = kwargs['shape'][-1].copy()

    # Remove channels dimension
    if len(kwargs['shape'][-1]) > 1:
        kwargs['module'].add_module(f"pre_reshape_{kwargs['i']}", Reshape([-1]))

    # Number of features can be defined by either a factor of the output size or explicitly
    if 'factor' in layer:
        out_features = max(1, int(np.prod(kwargs['out_shape']) * layer['factor']))
    else:
        out_features = layer['features']

    linear_layer = nn.Linear(in_features=np.prod(in_shape), out_features=out_features)
    kwargs['module'].add_module(f"linear_{kwargs['i']}", linear_layer)

    # Optional layers
    optional_layer('dropout', kwargs, layer, nn.Dropout1d(layer['dropout']))
    optional_layer('batch_norm', kwargs, layer, nn.BatchNorm1d(out_features))
    optional_layer('activation', kwargs, layer, nn.SELU())

    # Add channels dimension equal to input channels if input contains channels
    if len(kwargs['shape'][-1]) > 1:
        out_shape = [in_shape[0], int(out_features / in_shape[0])]
        kwargs['module'].add_module(f"post_reshape_{kwargs['i']}", Reshape(out_shape))
    else:
        out_shape = [out_features]

    kwargs['shape'].append(out_shape)


def sample(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Generates mean and standard deviation and randomly samples from a Gaussian distribution
    for a variational autoencoder

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
        factor : float, optional
            Output features is equal to the factor of the network's output,
            will be used if provided, else features will be used
        features : integer, optional
            Number of output features for the layer,
            if out_shape from kwargs and factor is provided, features will not be used
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    supported_params = ['factor', 'features']
    layer = check_layer(supported_params, kwargs, layer, check_params=check_params)

    in_shape = kwargs['shape'][-1].copy()

    # Remove channels dimension
    if len(kwargs['shape'][-1]) > 1:
        kwargs['module'].add_module(f"pre_reshape_{kwargs['i']}", Reshape([-1]))

    # Number of features can be defined by either a factor of the output size or explicitly
    try:
        out_features = max(1, int(np.prod(kwargs['out_shape']) * layer['factor']))
    except KeyError:
        out_features = layer['features']

    kwargs['module'].add_module(f"sample_{kwargs['i']}", Sample(np.prod(in_shape), out_features))

    # Add channels dimension equal to input channels if input contains channels
    if len(in_shape) > 1:
        out_shape = [in_shape[0], int(out_features / in_shape[0])]
        kwargs['module'].add_module(f"post_reshape_{kwargs['i']}", Reshape(out_shape))
    else:
        out_shape = [out_features]

    kwargs['shape'].append(out_shape)


def upsample(kwargs: dict, layer: dict, check_params: bool = True):
    """
    Constructs a 2x upscaler using linear upsampling

    Parameters
    ----------
    kwargs : dictionary
        i : integer
            Layer number
        shape : list[integer]
            Shape of the outputs from each layer
        module : Sequential
            Sequential module to contain the layer
    layer : dictionary
        For compatibility
    check_params : boolean, default = True
        If layer arguments should be checked if they are valid
    """
    check_layer([], kwargs, layer, check_params=check_params)

    kwargs['shape'].append(kwargs['shape'][-1].copy())
    kwargs['module'].add_module(
        f"upsample_{kwargs['i']}",
        nn.Upsample(scale_factor=2, mode='linear')
    )
    # Data size doubles
    kwargs['shape'][-1][1:] = [length * 2 for length in kwargs['shape'][-1][1:]]
