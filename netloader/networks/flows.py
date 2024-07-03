"""
Classes that contain multiple types of networks
"""
from typing import Any

import torch
import numpy as np
from torch import Tensor, nn
from torch.utils.data import DataLoader
from zuko.flows import NSF
from numpy import ndarray

from netloader.network import Network
from netloader.networks.base import BaseNetwork
from netloader.networks.encoder_decoder import Encoder
from netloader.utils.utils import get_device, label_change


class NormFlow(BaseNetwork):
    """
    Transforms a simple distribution into a distribution that reflects the input data

    Attributes
    ----------
    save_path : str
        Path to the network save file
    net : Network
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
    batch_predict(data, samples[1e3]) -> Tensor
        Generates probability distributions for the data batch
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

    Attributes
    ----------
    save_path : str
        Path to the network save file
    net : Network
        normalizing flow to predict low-dimensional data distribution
    encoder : Network
        Network to condition the normalizing flow from high-dimensional data
    encoder_loss : float, default = 0
        Loss for the output of the network
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
    train()
        Flips the train/eval state of the network and flow
    load(states_dir, load_num) -> ndarray | None
        Loads the flow and network from a previously saved state, if load_num != 0
    predict(loader, path=None, samples=[1e3]) -> tuple[ndarray, ndarray, ndarray]
        Generates probability distributions for a dataset and can save to a file
    batch_predict(data, samples=[1e3]) -> Tensor
        Generates probability distributions for the given data batch
    """
    def __init__(
            self,
            save_num: int,
            states_dir: str,
            net: Network,
            encoder: Network,
            mix_precision: bool = False,
            flow_checkpoint: int | None = None,
            description: str = '',
            verbose: str = 'epoch',
            train_epochs: tuple[int, int] | None = None,
            classes: Tensor | None = None):
        """
        Parameters
        ----------
        save_num : int
            File number to save the flow
        states_dir : str
            Directory to save the network and flow
        net : Network
            normalizing flow to predict low-dimensional data distribution
        encoder : Network
            Network to condition the normalizing flow from high-dimensional data
        mix_precision: bool, default = False
            If mixed precision should be used
        flow_checkpoint : int, default = None
            Network checkpoint to pass into the flow, if none, will use output from the network
        description : str, default = ''
            Description of the network training
        verbose : {'full', 'progress', None}
            If details about epoch should be printed ('full'), just a progress bar ('progress'),
            or nothing (None)
        classes : Tensor, default = None
            Unique classes of size C if using class classification
        """
        super().__init__(
            save_num,
            states_dir,
            net,
            mix_precision=mix_precision,
            description=description,
            verbose=verbose,
            classes=classes,
        )
        self._train_flow: bool
        self._train_encoder: bool
        self._checkpoint: int | None = flow_checkpoint
        self._epochs: tuple[int, int] | None = train_epochs
        self.flow_loss: float = 1
        self.encoder_loss: float = 1
        self.encoder: Network = encoder

        if self._epochs is None:
            self._epochs = (0, -1)
            assert self._epochs is not None

        self._train_flow = not self._epochs[0]
        self._train_encoder = bool(self._epochs[-1])

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
        output: Tensor
        flow_input: Tensor
        loss: Tensor = torch.tensor(0.).to(self._device)

        # Encoder outputs
        output = self.encoder(in_data)

        if self._checkpoint:
            flow_input = self.encoder.checkpoints[self._checkpoint]
        else:
            flow_input = output

        # normalizing flow loss
        if self.flow_loss:
            loss += self.flow_loss * -self.net(flow_input).log_prob(target).mean()

        # Default shape is (N, L), but cross entropy expects (N)
        if (self.encoder_loss and isinstance(self._loss_function, nn.CrossEntropyLoss) and
                self.classes is not None):
            target = label_change(target.squeeze(), self.classes)

        if self.encoder_loss:
            loss += self.encoder_loss * self._loss_function(output, target)

        if not self._train_state:
            self.encoder.layer_num = None
            return loss.item()

        # Update network
        self.net.optimiser.zero_grad()
        self.encoder.optimiser.zero_grad()
        loss.backward()

        if self._train_encoder:
            self.encoder.optimiser.step()

        if self._train_flow:
            self.net.optimiser.step()

        return loss.item()

    def train(self, train: bool) -> None:
        """
        Sets the train/eval state of the network/flow

        Parameters
        ----------
        train : bool
            If the network/flow should be in the train state
        """
        super().train(train)

        if self._train_state:
            self.encoder.train()
        else:
            self.encoder.eval()

    def _update_epoch(self) -> None:
        """
        Updates network and flow epoch if they are being trained
        """
        super()._update_epoch()
        assert self._epochs is not None

        if self._epoch >= self._epochs[0] != -1:
            self._train_flow = True

        if self._epoch >= self._epochs[-1] != -1:
            self._train_encoder = False

    def _scheduler(self) -> None:
        """
        Updates the scheduler for the flow and/or network if they are being trained
        """
        if self._train_flow:
            super()._scheduler()

        if self._train_encoder:
            self._update_scheduler(self.encoder.scheduler)

    def predict(
            self,
            loader: DataLoader,
            bin_num: int = 100,
            path: str | None = None,
            num: list[int] | None = None,
            header: list[str] | None = None,
            **_: Any) -> dict[str, ndarray]:
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
            **_: Any) -> tuple[ndarray, ...]:
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
        tuple[(N,S) ndarray, ...]
            S samples from N probability distributions for the given data
        """
        output: Tensor
        samples: Tensor
        flow_input: Tensor

        if num is None:
            num = [int(1e3)]

        # Encoder outputs
        output = self.encoder(data)

        if self._checkpoint:
            flow_input = self.encoder.checkpoints[self._checkpoint]
        else:
            flow_input = output

        # Generate samples
        samples = torch.transpose(
            self.net(flow_input).sample(num).squeeze(-1),
            0,
            1,
        )

        return output.detach().cpu().numpy(), samples.detach().cpu().numpy()


def norm_flow(
        features: int,
        transforms: int,
        learning_rate: float,
        hidden_features: list[int],
        context: int = 0) -> Network:
    """
    Generates a neural spline flow (NSF) for use in BaseNetwork

    Adds attributes of name ('flow'), optimiser (Adam), and scheduler (ReduceLROnPlateau)

    Parameters
    ----------
    features : int
        Dimensions of the probability distribution
    transforms : int
        Number of transforms
    learning_rate : float
        Learning rate of the NSF
    hidden_features : list[int]
        Number of features in each of the hidden layers
    context : int, default = 0
        Number of features to condition the NSF

    Returns
    -------
    Network
        Neural spline flow with attributes required for training
    """
    flow: NSF | Network = NSF(
        features=features,
        context=context,
        transforms=transforms,
        hidden_features=hidden_features,
    ).to(get_device()[1])

    flow.name = 'flow'
    flow.optimiser = torch.optim.Adam(flow.parameters(), lr=learning_rate)
    flow.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        flow.optimiser,
        patience=5,
        factor=0.5,
        verbose=True,
    )
    return flow
