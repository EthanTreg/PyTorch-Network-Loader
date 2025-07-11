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
from netloader.transforms import BaseTransform
from netloader.networks.encoder_decoder import Encoder


class NormFlow(BaseNetwork):
    """
    Transforms a simple distribution into a distribution that reflects the input data

    Requires last layer to be a normalizing flow and will not pass input data through the network

    Attributes
    ----------
    net : Module | Network
        Neural spline flow
    save_path : str, default = ''
        Path to the network save file
    description : str, default = ''
        Description of the network training
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current network training and validation losses
    transforms : dict[str, BaseTransform | None], default = {...: None, ...}
        Keys for the output data from predict and corresponding transforms
    idxs: (N) ndarray, default = None
        Data indices for random training & validation datasets
    optimiser : Optimizer, default = AdamW
        Network optimiser
    scheduler : LRScheduler, default = ReduceLROnPlateau
        Optimiser scheduler

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
            *_: Any,
            path: str | None = None,
            num: list[int] | None = None,
            **__: Any) -> dict[str, ndarray]:
        """
        Generates probability distributions for a dataset and can save to a file

        Parameters
        ----------
        path : str, default = None
            Path as CSV file to save the predictions if they should be saved
        num : list[int], default = [1e3]
            Number of samples to generate

        Returns
        -------
        dict[str, ndarray]
            Predicted distribution
        """
        data: dict[str, ndarray]
        samples: ndarray
        transform: BaseTransform | None = self.transforms['preds']

        if num is None:
            num = [int(1e3)]

        samples = self.net().sample(num).moveaxis(0, -1).detach().cpu().numpy()
        data = {'samples': samples if transform is None else transform(samples, back=True)}
        self._save_predictions(path, data)
        return data


class NormFlowEncoder(Encoder):
    """
    Calculates the loss for a network and normalizing flow that takes high-dimensional data
    and predicts a low-dimensional data distribution

    Requires the normalizing flow to be the last layer in the network

    Attributes
    ----------
    net : Module | Network
        normalizing flow to predict low-dimensional data distribution
    flow_loss : float, default = 1
        Loss weight for the normalizing flow
    encoder_loss : float, default = 1
        Loss weight for the output of the encoder
    save_path : str, default = ''
        Path to the network save file
    description : str, default = ''
        Description of the network training
    losses : tuple[list[Tensor], list[Tensor]], default = ([], [])
        Current flow training and validation losses
    transforms : dict[str, BaseTransform | None], default = {...: None, ...}
        Keys for the output data from predict and corresponding transforms
    idxs: (N) ndarray, default = None
        Data indices for random training & validation datasets
    classes : (C) Tensor, default = None
        Unique classes of size C if using class classification
    optimiser : Optimizer, default = AdamW
        Network optimiser
    scheduler : LRScheduler, default = ReduceLROnPlateau
        Optimiser scheduler

    Methods
    -------
    set_optimiser(*args, **kwargs)
        Sets the optimiser for the network
    predict(loader, bin_num=100, path=None, num=[1e3], header=[...]) -> dict[str, ndarray]
        Generates probability distributions for a dataset and can save to a file
    batch_predict(data, num=[1e3]) -> tuple[ndarray | list[None], ndarray | list[None]]
        Generates probability distributions for the given data batch
    extra_repr() -> str
        Displays layer parameters when printing the architecture
    """
    def __init__(
            self,
            save_num: int | str,
            states_dir: str,
            net: torch.nn.Module | Network,
            overwrite: bool = False,
            mix_precision: bool = False,
            net_checkpoint: int | None = None,
            description: str = '',
            verbose: str = 'epoch',
            train_epochs: tuple[int, int] = (0, -1),
            learning_rate: tuple[float, float] = (1e-3,) * 2,
            classes: Tensor | None = None,
            transform: BaseTransform | None = None,
            in_transform: BaseTransform | None = None,
            optimiser_kwargs: dict[str, Any] | None = None,
            scheduler_kwargs: dict[str, Any] | None = None) -> None:
        """
        Parameters
        ----------
        save_num : int | str
            File number or name to save the flow
        states_dir : str
            Directory to save the network and flow
        net : nn.Module | Network
            normalizing flow to predict low-dimensional data distribution
        overwrite : bool, default = False
            If saving can overwrite an existing save file, if True and file with the same name
            exists, an error will be raised
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
        in_transform : BaseTransform, default = None
            Transformation for the input data
        optimiser_kwargs : dict[str, Any] | None, default = None
            Optional keyword arguments to pass to set_optimiser
        scheduler_kwargs : dict[str, Any] | None, default = None
            Optional keyword arguments to pass to set_scheduler
        """
        super().__init__(
            save_num,
            states_dir,
            net,
            overwrite=overwrite,
            mix_precision=mix_precision,
            learning_rate=0,
            description=description,
            verbose=verbose,
            classes=classes,
            transform=transform,
            in_transform=in_transform,
            optimiser_kwargs=optimiser_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        self._train_flow: bool
        self._train_encoder: bool
        self._checkpoint: int | None = net_checkpoint
        self._epochs: tuple[int, int] = train_epochs
        self.flow_loss: float = 1
        self.encoder_loss: float = 1

        self._train_flow = not self._epochs[0]
        self._train_encoder = bool(self._epochs[-1])
        self.transforms |= {
            'distributions': transform,
            'probs': None,
            'max': transform,
            'meds': transform,
        }

        if not self._train_encoder:
            self.net.net[:-1].requires_grad_(False)

        if not self._train_flow:
            self.net.net[-1].requires_grad_(False)

        self.optimiser = self.set_optimiser([
            {'encoder_params': self.net.net[:-1].parameters(), 'lr': learning_rate[0]},
            {'flow_params': self.net.net[-1].parameters(), 'lr': learning_rate[1]},
        ], **optimiser_kwargs or {})
        self.scheduler = self.set_scheduler(self.optimiser, **scheduler_kwargs or {})

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.load_state_dict(
                {'factor': 0.5, 'min_lr': min(learning_rate) * 1e-3} | (scheduler_kwargs or {}),
            )

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'train_flow': self._train_flow,
            'train_encoder': self._train_encoder,
            'checkpoint': self._checkpoint,
            'epochs': self._epochs,
            'flow_loss': self.flow_loss,
            'encoder_loss': self.encoder_loss,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._train_flow = state['train_flow']
        self._train_encoder = state['train_encoder']
        self._checkpoint = state['checkpoint']
        self._epochs = state['epochs']
        self.flow_loss = state['flow_loss']
        self.encoder_loss = state['encoder_loss']

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
            loss += self.encoder_loss * self._loss_func(output, target)

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
            inputs: bool = False,
            bin_num: int = 100,
            path: str | None = None,
            num: list[int] | None = None,
            **_: Any) -> dict[str, ndarray]:
        """
        Generates probability distributions for a dataset and can save to a file

        Parameters
        ----------
        loader : DataLoader
            Dataset to generate predictions for
        inputs : bool, default = False,
            If the input data should be returned and saved
        bin_num : int, default = 100
            Number of bins for calculating the probability of the target and maximum of the
            distribution, higher is more precise but requires more samples
        path : str, default = None
            Path as a pkl file to save the predictions if they should be saved
        num : list[int], default = [1e3]
            Number of samples to generate from the predicted distribution

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

        data = super().predict(loader, inputs=inputs, num=num)

        if data['distributions'] is None:
            self._save_predictions(path, data)
            return data

        for target, distribution in zip(data['targets'], data['distributions']):
            hist, bins = np.histogram(distribution, bins=bin_num, density=True)
            prob = hist * (bins[1] - bins[0])
            bins[-1] += 1e-6
            probs.append(prob[np.clip(np.digitize(target, bins) - 1, 0, bin_num - 1)])
            maxima.append(bins[np.argmax(hist)])

        data['probs'] = np.stack(probs)
        data['max'] = np.stack(maxima)
        data['meds'] = np.median(data['distributions'], axis=-1)
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

    def extra_repr(self) -> str:
        """
        Displays architecture parameters when printing the network

        Returns
        -------
        str
            Architecture parameters
        """
        return (f'{super().extra_repr()}, '
                f'train_epochs: {self._epochs}, '
                f'flow_weight: {self.flow_loss}, '
                f'encoder_weight: {self.encoder_loss}')
