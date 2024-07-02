"""
Classes for encoder, decoder, or autoencoder type architectures
"""
from typing import Any

import torch
import numpy as np
from torch import Tensor, nn

from netloader.network import Network
from netloader.utils.utils import label_change
from netloader.networks.base import BaseNetwork


class Autoencoder(BaseNetwork):
    """
    Network handler for autoencoder type networks

    Attributes
    ----------
    save_path : str
        Path to the network save file
    net : Network
        Autoencoder network
    reconstruct_loss : float, default = 1
        Loss weight for the reconstruction MSE loss
    latent_loss : float, default = 1e-2
        Loss weight for the latent MSE loss
    bound_loss : float, default = 1e-3
        Loss weight for the latent bounds loss
    description : str, default = ''
        Description of the network training
    transform : tuple[float, float], default = None
        Expected label transformation of the network, with the transformation
        (data - transform[0]) / transform[1]
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current autoencoder training and validation losses
    idxs: ndarray, default = None
        Data indices for random training & validation datasets
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            net: Network,
            description: str = '',
            verbose: str = 'full'):
        """
        Parameters
        ----------
        save_num : int
            File number to save the network
        states_dir : str
            Directory to save the network
        net : Network
            Network to predict low-dimensional data
        description : str, default = ''
            Description of the network training
        verbose : {'full', 'progress', None}
            If details about epoch should be printed ('full'), just a progress bar ('progress'),
            or nothing (None)
        """
        super().__init__(save_num, states_dir, net, description=description, verbose=verbose)
        self.reconstruct_loss: float = 1
        self.latent_loss: float = 1e-2
        self.bound_loss: float = 1e-3

    def _loss(self, in_data: Tensor, target: Tensor) -> float:
        """
        Calculates the loss from the autoencoder's predictions

        Parameters
        ----------
        in_data : (N,...) Tensor
            Input high dimensional data of batch size N and the remaining dimensions depend on the
            network used
        target : (N,...) Tensor
            Latent target low dimensional data of batch size N and the remaining dimensions depend
            on the network used

        Returns
        -------
        float
            Loss from the autoencoder's predictions'
        """
        loss: Tensor
        bounds: Tensor = torch.tensor([0., 1.]).to(self._device)
        output: Tensor = self.net(in_data)
        latent: Tensor = self.net.checkpoints[-1]

        loss = self.reconstruct_loss * nn.MSELoss()(output, in_data) + self.net.kl_loss

        if self.latent_loss:
            loss += self.latent_loss * nn.MSELoss()(latent, target)

        if self.bound_loss:
            loss += self.bound_loss * torch.mean(torch.cat((
                (bounds[0] - latent) ** 2 * (latent < bounds[0]),
                (latent - bounds[1]) ** 2 * (latent > bounds[1]),
            )))

        self._update(loss)
        return loss.item()


class Decoder(BaseNetwork):
    """
    Calculates the loss for a network that takes low-dimensional data and predicts
    high-dimensional data

    Attributes
    ----------
    save_path : str
        Path to the network save file
    net : Network
        Neural network
    description : str, default = ''
        Description of the network training
    transform : tuple[float, float], default = None
        Expected low-dimensional data transformation of the network, with the transformation
        (data - transform[0]) / transform[1]
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current network training and validation losses
    idxs: ndarray, default = None
        Data indices for random training & validation datasets
    """
    def _data_loader_translation(self, low_dim: Tensor, high_dim: Tensor) -> tuple[Tensor, Tensor]:
        """
        Takes the low and high dimensional tensors from the data loader, and orders them as inputs
        and targets for the network

        Parameters
        ----------
        low_dim : Tensor
            Low dimensional tensor from the data loader
        high_dim : Tensor
            High dimensional tensor from the data loader

        Returns
        -------
        tuple[Tensor, Tensor]
            Input and output target tensors
        """
        return low_dim, high_dim

    def _loss(self, in_data: Tensor, target: Tensor) -> float:
        """
        Calculates the loss from the network's predictions'

        Parameters
        ----------
        in_data : (N,...) Tensor
            Input low dimensional data of batch size N and the remaining dimensions depend on the
            network used
        target : (N, ...) Tensor
            Target high dimensional data of batch size N and the remaining dimensions depend on the
            network used

        Returns
        -------
        float
            Loss from the network's predictions
        """
        output: Tensor = self.net(in_data)
        loss: Tensor = nn.MSELoss()(output, target)
        self._update(loss)
        return loss.item()


class Encoder(BaseNetwork):
    """
    Calculates the loss for a network that takes high-dimensional data
    and predicts low-dimensional data

    Attributes
    ----------
    save_path : str
        Path to the network save file
    net : Network
        Neural network
    description : str, default = ''
        Description of the network training
    transform : tuple[float, float], default = None
        Expected low-dimensional data transformation of the network, with the transformation
        (data - transform[0]) / transform[1]
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current network training and validation losses
    idxs: ndarray, default = None
        Data indices for random training & validation datasets
    classes : (C) Tensor, default = None
        Unique classes of size C if using class classification

    Methods
    -------
    batch_predict(high_dim) -> Tensor
        Generates predictions for the given data
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            net: Network,
            description: str = '',
            verbose: str = 'full',
            classes: Tensor | None = None):
        """
        Parameters
        ----------
        save_num : int
            File number to save the network
        states_dir : str
            Directory to save the network
        net : Network
            Network to predict low-dimensional data
        description : str, default = ''
            Description of the network training
        verbose : {'full', 'progress', None}
            If details about epoch should be printed ('full'), just a progress bar ('progress'),
            or nothing (None)
        classes : (C) Tensor, default = None
            Unique classes of size C if using class classification
        """
        super().__init__(save_num, states_dir, net, description=description, verbose=verbose)
        self._loss_function: nn.MSELoss | nn.CrossEntropyLoss
        self.classes: Tensor | None

        if classes is None:
            self.classes = classes
            self._loss_function = nn.MSELoss()
        else:
            self.classes = classes.to(self._device)
            self._loss_function = nn.CrossEntropyLoss()

    def _loss(self, in_data: Tensor, target: Tensor) -> float:
        """
        Calculates the loss from the network's predictions'

        Parameters
        ----------
        in_data : (N,...) Tensor
            Input high dimensional data of batch size N and the remaining dimensions depend on the
            network used
        target : (N, ...) Tensor
            Target low dimensional data of batch size N and the remaining dimensions depend on the
            network used

        Returns
        -------
        float
            Loss from the network's predictions'
        """
        loss: Tensor
        output: Tensor = self.net(in_data)

        # Default shape is (N, L), but cross entropy expects (N)
        if isinstance(self._loss_function, nn.CrossEntropyLoss) and self.classes is not None:
            target = label_change(target.squeeze(), self.classes)

        loss = self._loss_function(output, target)
        self._update(loss)
        return loss.item()

    def batch_predict(self, data: Tensor, **_: Any) -> tuple[np.ndarray, ...]:
        """
        Generates predictions for the given data

        Parameters
        ----------
        data : Tensor
            Data to generate predictions for

        Returns
        -------
        tuple[N ndarray, ...]
            N predictions for the given data
        """
        output: np.ndarray = super().batch_predict(data)[0]

        if isinstance(self._loss_function, nn.CrossEntropyLoss):
            output = np.argmax(output, axis=-1)

        return (output,)
