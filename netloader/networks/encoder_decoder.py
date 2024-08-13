"""
Classes for encoder, decoder, or autoencoder type architectures
"""
from typing import Any

import torch
import numpy as np
from numpy import ndarray
from torch import Tensor, nn

from netloader.network import Network
from netloader.utils.utils import label_change
from netloader.networks.base import BaseNetwork
from netloader.utils.transforms import BaseTransform


class Autoencoder(BaseNetwork):
    """
    Network handler for autoencoder type networks

    Attributes
    ----------
    save_path : str
        Path to the network save file
    optimiser : Optimizer
        Network optimiser, uses AdamW optimiser
    scheduler : LRScheduler
        Optimiser scheduler, uses reduce learning rate on plateau
    net : Module | Network
        Autoencoder network
    reconstruct_loss : float, default = 1
        Loss weight for the reconstruction MSE loss
    latent_loss : float, default = 1e-2
        Loss weight for the latent MSE loss
    bound_loss : float, default = 1e-3
        Loss weight for the latent bounds loss
    kl_loss : float, default = 1e-1
        Relative weight if performing a KL divergence loss on the latent space
    description : str, default = ''
        Description of the network training
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current autoencoder training and validation losses
    idxs: ndarray, default = None
        Data indices for random training & validation datasets
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            net: nn.Module | Network,
            mix_precision: bool = False,
            learning_rate: float = 1e-3,
            description: str = '',
            verbose: str = 'epoch',
            reconstruct_loss: nn.Module = nn.MSELoss(),
            transform: BaseTransform | None = None,
            latent_transform: BaseTransform | None = None):
        """
        Parameters
        ----------
        save_num : int
            File number to save the network
        states_dir : str
            Directory to save the network
        net : Module | Network
            Network to predict low-dimensional data
        mix_precision: bool, default = False
            If mixed precision should be used
        learning_rate : float, default = 1e-3
            Optimiser initial learning rate, if None, no optimiser or scheduler will be set
        description : str, default = ''
            Description of the network training
        verbose : {'full', 'progress', None}
            If details about epoch should be printed ('full'), just a progress bar ('progress'),
            or nothing (None)
        reconstruct_loss : nn.Module, default = MSELoss()
            Which loss function to use for the reconstruction loss
        transform : BaseTransform, default = None
            Transformation applied to the input data
        latent_transform : BaseTransform, default = None
            Transformation applied to the latent space
        """
        super().__init__(
            save_num,
            states_dir,
            net,
            mix_precision=mix_precision,
            learning_rate=learning_rate,
            description=description,
            verbose=verbose,
            transform=transform,
        )
        self._loss_function: nn.Module = reconstruct_loss
        self.reconstruct_loss: float = 1
        self.latent_loss: float = 1e-2
        self.bound_loss: float = 1e-3
        self.kl_loss: float = 1e-1

        self._header['latent'] = latent_transform

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
        latent: Tensor | None = None
        bounds: Tensor = torch.tensor([0., 1.]).to(self._device)
        output: Tensor = self.net(in_data)

        if self.net.checkpoints:
            latent = self.net.checkpoints[-1]

        loss = self.reconstruct_loss * self._loss_function(output, in_data)

        if self.latent_loss and latent is not None:
            loss += self.latent_loss * nn.MSELoss()(latent, target)

        if self.bound_loss and latent is not None:
            loss += self.bound_loss * torch.mean(torch.cat((
                (bounds[0] - latent) ** 2 * (latent < bounds[0]),
                (latent - bounds[1]) ** 2 * (latent > bounds[1]),
            )))

        if self.kl_loss:
            loss += self.kl_loss * self.net.kl_loss

        self._update(loss)
        return loss.item()

    def batch_predict(self, data: Tensor, **_: Any) -> tuple[ndarray, ...]:
        """
        Generates predictions for the given data batch

        Parameters
        ----------
        data : (N,...) Tensor
            N data to generate predictions for

        Returns
        -------
        tuple[(N,...) ndarray, ...]
            N predictions for the given data
        """
        return (
            self.net(data).detach().cpu().numpy(),
            self.net.checkpoints[-1].detach().cpu().numpy(),
        )


class Decoder(BaseNetwork):
    """
    Calculates the loss for a network that takes low-dimensional data and predicts
    high-dimensional data

    Attributes
    ----------
    save_path : str
        Path to the network save file
    optimiser : Optimizer
        Network optimiser, uses AdamW optimiser
    scheduler : LRScheduler
        Optimiser scheduler, uses reduce learning rate on plateau
    net : Module | Network
        Neural network
    description : str, default = ''
        Description of the network training
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
    optimiser : Optimizer
        Network optimiser, uses AdamW optimiser
    scheduler : LRScheduler
        Optimiser scheduler, uses reduce learning rate on plateau
    net : Module | Network
        Neural network
    description : str, default = ''
        Description of the network training
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
            net: nn.Module | Network,
            mix_precision: bool = False,
            learning_rate: float = 1e-3,
            description: str = '',
            verbose: str = 'epoch',
            classes: Tensor | None = None,
            transform: BaseTransform | None = None):
        """
        Parameters
        ----------
        save_num : int
            File number to save the network
        states_dir : str
            Directory to save the network
        net : nn.Module | Network
            Network to predict low-dimensional data
        mix_precision: bool, default = False
            If mixed precision should be used
        learning_rate : float, default = 1e-3
            Optimiser initial learning rate, if None, no optimiser or scheduler will be set
        description : str, default = ''
            Description of the network training
        verbose : {'full', 'progress', None}
            If details about epoch should be printed ('full'), just a progress bar ('progress'),
            or nothing (None)
        classes : (C) Tensor, default = None
            Unique classes of size C if using class classification
        transform : BaseTransform, default = None
            Transformation of the low-dimensional data
        """
        super().__init__(
            save_num,
            states_dir,
            net,
            mix_precision=mix_precision,
            learning_rate=learning_rate,
            description=description,
            verbose=verbose,
            transform=transform,
        )
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
        if self.classes is not None:
            target = label_change(target.squeeze(), self.classes)

        loss = self._loss_function(output, target)
        self._update(loss)
        return loss.item()

    def batch_predict(self, data: Tensor, **_: Any) -> tuple[ndarray, ...]:
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
        output: ndarray = super().batch_predict(data)[0]

        if isinstance(self._loss_function, nn.CrossEntropyLoss):
            output = np.argmax(output, axis=-1, keepdims=True)

        return (output,)
