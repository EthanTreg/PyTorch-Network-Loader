"""
Classes that contain multiple types of networks
"""
from typing import Any

import torch
import numpy as np
from torch import optim, Tensor
from torch.utils.data import DataLoader
from zuko.distributions import NormalizingFlow
from numpy import ndarray

from netloader.network import Network
from netloader.utils.utils import label_change
from netloader.networks.base import BaseNetwork
from netloader.networks.encoder_decoder import Encoder


class NormFlow(BaseNetwork):
    """
    Transforms a simple distribution into a distribution that reflects the input data

    Requires last layer to be a normalizing flow and will not pass input data through the network

    Attributes
    ----------
    save_path : str
        Path to the network save file
    optimiser : Optimizer
        Network optimiser, uses AdamW optimiser
    scheduler : LRScheduler
        Optimiser scheduler, uses reduce learning rate on plateau
    net : Module | Network
        Neural spline flow
    description : str, default = ''
        Description of the network training
    transform : tuple[float, float], default = None
        Expected low-dimensional data transformation of the network, with the transformation
        (data - transform[0]) / transform[1]
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current network training and validation losses
    idxs: ndarray, default = None
        Data indices for random training & validation datasets

    Methods
    -------
    predict(path=None, num=[1e3], header=[...]) -> dict[str, ndarray]
        Generates probability distributions for a dataset and can save to a file
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

    def _loss(self, _: Any, target: Tensor) -> float:
        """
        Calculates the loss from the flow's predictions

        Parameters
        ----------
        target : (N, ...) Tensor
            Target high dimensional data of batch size N and the remaining dimensions depend on the
            network used

        Returns
        -------
        float
            Loss from the network's predictions
        """
        loss: Tensor = -self.net().log_prob(target).mean()
        self._update(loss)
        return loss.item()

    def predict(
            self,
            path: str | None = None,
            num: list[int] | None = None,
            header: list[str] | None = None,
            **_: Any) -> dict[str, ndarray]:
        """
        Generates probability distributions for a dataset and can save to a file

        Parameters
        ----------
        path : str, default = None
            Path as CSV file to save the predictions if they should be saved
        num : list[int], default = [1e3]
            Number of samples to generate
        header : list[str], default = ['samples']
            Header for the predicted data, only used by child classes

        Returns
        -------
        dict[str, ndarray]
            Predicted distribution
        """
        data: dict[str, ndarray]

        if header is None:
            header = ['samples']

        if num is None:
            num = [int(1e3)]

        data = {header[0]: self.net().sample(num).moveaxis(0, -1).detach().cpu().numpy()}
        self._save_predictions(path, data)
        return data


class NormFlowEncoder(Encoder):
    """
    Calculates the loss for a network and normalizing flow that takes high-dimensional data
    and predicts a low-dimensional data distribution

    Requires the normalizing flow to be the last layer in the network

    Attributes
    ----------
    save_path : str
        Path to the network save file
    optimiser : Optimizer
        Network optimiser, uses AdamW optimiser
    scheduler : LRScheduler
        Optimiser scheduler, uses reduce learning rate on plateau
    net : Module | Network
        normalizing flow to predict low-dimensional data distribution
    flow_loss : float, default = 1
        Loss weight for the normalizing flow
    encoder_loss : float, default = 1
        Loss weight for the output of the encoder
    description : str, default = ''
        Description of the network training
    transform : tuple[float, float], default = None
        Expected low-dimensional data transformation of the network, with the transformation
        (data - transform[0]) / transform[1]
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current flow training and validation losses
    idxs: ndarray, default = None
        Data indices for random training & validation datasets
    classes : (C) Tensor, default = None
        Unique classes of size C if using class classification

    Methods
    -------
    predict(loader, bin_num=100, path=None, num=[1e3], header=[...]) -> dict[str, ndarray]
        Generates probability distributions for a dataset and can save to a file
    batch_predict(data, num=[1e3]) -> tuple[ndarray | list[None], ndarray | list[None]]
        Generates probability distributions for the given data batch
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            net: torch.nn.Module | Network,
            mix_precision: bool = False,
            net_checkpoint: int | None = None,
            description: str = '',
            verbose: str = 'epoch',
            train_epochs: tuple[int, int] | None = None,
            learning_rate: tuple[float, float] | None = None,
            classes: Tensor | None = None):
        """
        Parameters
        ----------
        save_num : int
            File number to save the flow
        states_dir : str
            Directory to save the network and flow
        net : nn.Module | Network
            normalizing flow to predict low-dimensional data distribution
        mix_precision: bool, default = False
            If mixed precision should be used
        net_checkpoint : int, default = None
            Network checkpoint for calculating the encoder's loss, if none, will use output from the
            network if output is a Tensor, else no encoder loss will be used
        description : str, default = ''
            Description of the network training
        verbose : {'full', 'progress', None}
            If details about epoch should be printed ('full'), just a progress bar ('progress'),
            or nothing (None)
        train_epochs : tuple[int, int], default = (0, -1)
            Epoch when to start training the normalizing flow and epoch when to stop training the
            encoder, if both are zero, then flow will be trained from the beginning and encoder will
            not be trained, if both are -1, then flow will never be trained and encoder will always
            be trained
        learning_rate : tuple[float, float], default = (1e-3, 1e-3)
            Optimiser initial learning rate for encoder and normalizing flow, if None, no optimiser
            or scheduler will be set
        classes : Tensor, default = None
            Unique classes of size C if using class classification
        """
        super().__init__(
            save_num,
            states_dir,
            net,
            mix_precision=mix_precision,
            learning_rate=0,
            description=description,
            verbose=verbose,
            classes=classes,
        )
        self._train_flow: bool
        self._train_encoder: bool
        self._checkpoint: int | None = net_checkpoint
        self._epochs: tuple[int, int] = train_epochs if train_epochs is not None else (0, -1)
        self.flow_loss: float = 1
        self.encoder_loss: float = 1

        self._train_flow = not self._epochs[0]
        self._train_encoder = bool(self._epochs[-1])

        if not self._train_flow:
            self.net.net[-1].requires_grad_(False)

        if not self._train_encoder:
            self.net.net[:-1].requires_grad_(False)

        if learning_rate is None:
            learning_rate = (1e-3,) * 2

        if learning_rate:
            self.optimiser = optim.AdamW([
                {'params': self.net.net[:-1].parameters(), 'lr': learning_rate[0]},
                {'params': self.net.net[-1].parameters(), 'lr': learning_rate[1]},
            ])
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, factor=0.5)

    def _loss(self, in_data: Tensor, target: Tensor) -> float:
        """
        Calculates the loss from the network and flow's predictions

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
            Loss from the flow's predictions'
        """
        loss: Tensor = torch.tensor(0.).to(self._device)
        output: Tensor | NormalizingFlow

        # Encoder outputs
        output = self.net(in_data)

        if isinstance(output, NormalizingFlow) and self.flow_loss:
            loss += self.flow_loss * -output.log_prob(target).mean()

        if isinstance(output, NormalizingFlow) and self._checkpoint:
            output = self.net.checkpoints[self._checkpoint]

        # Default shape is (N, L), but cross entropy expects (N)
        if self.encoder_loss and self.classes is not None and isinstance(output, Tensor):
            target = label_change(target.squeeze(), self.classes)

        if self.encoder_loss and isinstance(output, Tensor):
            loss += self.encoder_loss * self._loss_function(output, target)

        self._update(loss)
        return loss.item()

    def _update_epoch(self) -> None:
        """
        Updates network and flow epoch if they are being trained
        """
        super()._update_epoch()

        if not self._train_flow and self._epoch >= self._epochs[0] != -1:
            self._train_flow = True
            self.net.net[-1].requires_grad_(True)

        if self._train_encoder and self._epoch >= self._epochs[-1] != -1:
            self._train_encoder = False
            self.net.net[:-1].requires_grad_(False)

    def predict(
            self,
            loader: DataLoader,
            bin_num: int = 100,
            path: str | None = None,
            num: list[int] | None = None,
            header: list[str] | None = None,
            **_: Any) -> dict[str, ndarray]:
        # pylint: disable=line-too-long
        """
        Generates probability distributions for a dataset and can save to a file

        Parameters
        ----------
        loader : DataLoader
            Dataset to generate predictions for
        bin_num : int, default = 100
            Number of bins for calculating the probability of the target and maximum of the
            distribution, higher is more precise but requires more samples
        path : str, default = None
            Path as a pkl file to save the predictions if they should be saved
        num : list[int], default = [1e3]
            Number of samples to generate from the predicted distribution
        header : list[str], default = ['ids', 'targets', 'preds', 'distributions', 'probs', 'max', 'meds']
            Header for the predicted data, only used by child classes

        Returns
        -------
        dict[str, ndarray]
            Prediction IDs, target values, target probability, distribution maximum, distribution
            median, and predicted distribution with samples S for dataset of size N
        """
        probs: list[ndarray] = []
        maxima: list[ndarray] = []
        data: dict[str, ndarray]
        hist: ndarray
        bins: ndarray
        prob: ndarray

        if header is None:
            header = ['ids', 'targets', 'preds', 'distributions', 'probs', 'max', 'meds']

        data = super().predict(loader, header=header, num=num)

        if data['distributions'] is None:
            self._save_predictions(path, data)
            return data

        for target, distribution in zip(data['targets'], data['distributions']):
            hist, bins = np.histogram(distribution, bins=bin_num, density=True)
            prob = hist * (bins[1] - bins[0])
            bins[-1] += 1e-6
            probs.append(prob[np.clip(np.digitize(target, bins) - 1, 0, bin_num - 1)])
            maxima.append(bins[np.argmax(hist)])

        data[header[-3]] = np.stack(probs)
        data[header[-2]] = np.stack(maxima)
        data[header[-1]] = np.median(data['distributions'], axis=-1)
        self._save_predictions(path, data)
        return data

    def batch_predict(
            self,
            data: Tensor,
            num: list[int] | None = None,
            **_: Any) -> tuple[ndarray | list[None], ndarray | list[None]]:
        """
        Generates probability distributions for the data batch

        Parameters
        ----------
        data : (N,...) Tensor
            Data to generate distributions for
        num : list[int], default = [1e3]
            Number of samples to generate

        Returns
        -------
        tuple[(N,...) ndarray | list[None], (N,S) ndarray | list[None]]
            Network output and S samples from each probability distribution in batch of N
        """
        samples: ndarray | list[None] = [None]
        output: ndarray | Tensor | NormalizingFlow | list[None] = self.net(data)

        if num is None:
            num = [int(1e3)]

        # Generate samples
        if isinstance(output, NormalizingFlow):
            samples = torch.transpose(
                output.sample(num).squeeze(-1),
                0,
                1,
            ).detach().cpu().numpy()
            output = [None]
        else:
            assert isinstance(output, Tensor)
            output = output.detach().cpu().numpy()

        if output is None and self._checkpoint:
            output = self.net.checkpoints[self._checkpoint].detach().cpu().numpy()

        assert not isinstance(output, Tensor)
        return output, samples
