"""
Base network class to base other networks off
"""
import os
import pickle
import logging as log
from time import time
from typing import Any, Self, Union

import torch
import numpy as np
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from numpy import ndarray

from netloader.network import Network
from netloader.utils.transforms import BaseTransform
from netloader.utils.utils import save_name, get_device, progress_bar

Param = dict[Any, Union[torch.Tensor, 'Param']]


class BaseNetwork:
    """
    Base network class that other types of networks build from

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
        Description of the network
    losses : tuple[list[float], list[float]], default = ([], [])
        Network training and validation losses
    header : dict[str, BaseTransform | None], default = {...: None, ...}
        Keys for the output data from predict and corresponding transforms
    idxs: (N) ndarray, default = None
        Data indices for random training & validation datasets
    in_transform : BaseTransform, default = None
        Transformation for the input data

    Methods
    -------
    train(train)
        Flips the train/eval state of the network
    training(epoch, training)
        Trains & validates the network for each epoch
    save()
        If save_num is provided, saves the network to the states directory
    predict(loader, path=None, **kwargs) -> dict[str, (N,...) ndarray]
        Generates predictions for a dataset and can save to a file
    batch_predict(high_dim) -> tuple[(N,...) ndarray]
        Generates predictions for the given data batch
    to(*args, **kwargs) -> Self
        Move and/or cast the parameters and buffers
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
            transform: BaseTransform | None = None,
            in_transform: BaseTransform | None = None):
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
            Description of the network
        verbose : {'epoch', 'full', 'progress', None}
            If details about each epoch should be printed ('epoch'), details about epoch and epoch
            progress (full), just total progress ('progress'), or nothing (None)
        transform : BaseTransform, default = None
            Transformation of the network's output
        in_transform : BaseTransform, default = None
            Transformation for the input data
        """
        self._train_state: bool = True
        self._half: bool = mix_precision
        self._epoch: int = 0
        self._verbose: str = verbose
        self._device: torch.device = get_device()[1]
        self.save_path: str | None
        self.description: str = description
        self.losses: tuple[list[float], list[float]] = ([], [])
        self.header: dict[str, BaseTransform | None] = {
            'ids': None,
            'targets': transform,
            'preds': transform,
        }
        self.idxs: ndarray | None = None
        self.optimiser: optim.Optimizer
        self.scheduler: optim.lr_scheduler.LRScheduler
        self.net: nn.Module | Network = net
        self.in_transform: BaseTransform | None = in_transform

        if not isinstance(self.net, Network) and not hasattr(self.net, 'net'):
            self.net.net = nn.ModuleList(self.net._modules.values())
        elif not isinstance(self.net, Network) and not isinstance(self.net.net, nn.ModuleList):
            raise NameError('net requires an indexable module list as attribute net, but attribute '
                            'net already exists')

        if not hasattr(self.net, 'name'):
            self.net.name = type(self.net).__name__

        if save_num:
            self.save_path = save_name(save_num, states_dir, self.net.name)

            if os.path.exists(self.save_path):
                log.getLogger(__name__).warning(f'{self.save_path} already exists and will be'
                                                f'overwritten if training continues')
        else:
            self.save_path = None

        if learning_rate:
            self.optimiser = optim.AdamW(self.net.parameters(), lr=learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, factor=0.5)

    def _data_loader_translation(self, low_dim: Tensor, high_dim: Tensor) -> tuple[Tensor, Tensor]:
        """
        Takes the low and high dimensional tensors from the data loader, and orders them as inputs
        and targets for the network

        Parameters
        ----------
        low_dim : (N,...) Tensor
            Low dimensional tensor from the data loader for N data
        high_dim : (N,...) Tensor
            High dimensional tensor from the data loader for N data

        Returns
        -------
        tuple[(N,...) Tensor, (N,...) Tensor]
            Input and output target tensors for N data
        """
        return high_dim, low_dim

    def _train_val(self, loader: DataLoader) -> float:
        """
        Trains the network for one epoch

        Parameters
        ----------
        loader : DataLoader
            PyTorch DataLoader that contains data to train

        Returns
        -------
        float
            Average loss value
        """
        epoch_loss: float = 0
        low_dim: Tensor
        high_dim: Tensor

        with torch.set_grad_enabled(self._train_state), torch.autocast(
                enabled=self._half,
                device_type=self._device.type,
                dtype=torch.bfloat16):
            for i, (_, low_dim, high_dim, *_) in enumerate(loader):
                low_dim = low_dim.to(self._device)
                high_dim = high_dim.to(self._device)
                epoch_loss += self._loss(*self._data_loader_translation(low_dim, high_dim))
                self._update_scheduler(epoch=self._epoch + (i + 1) / len(loader))

                if self._verbose == 'full':
                    progress_bar(i, len(loader))

        return epoch_loss / len(loader)

    def _loss(self, in_data: Tensor, target: Tensor) -> float:
        """
        Empty method for child classes to base their loss functions on

        Parameters
        ----------
        in_data : (N,...) Tensor
            Input data of batch size N and the remaining dimensions depend on the network used
        target : (N,...) Tensor
            Target data of batch size N and the remaining dimensions depend on the network used

        Returns
        -------
        float
            Loss
        """

    def _param_device(self, params: Param, *args: Any, **kwargs: Any) -> None:  # pylint: disable=protected-access
        r"""
        Sends parameters to the device, such as the parameters in the optimiser

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self
        """
        for param in params.values():
            if isinstance(param, dict):
                self._param_device(param, *args, **kwargs)
            else:
                param.data = param.data.to(*args, **kwargs)

                if param._grad is not None:
                    param._grad.data = param._grad.data.to(*args, **kwargs)

    def _update(self, loss: Tensor) -> None:
        """
        Updates the network using backpropagation

        Parameters
        ----------
        loss : Tensor
            Loss to perform backpropagation from
        """
        if self._train_state:
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    def _update_scheduler(self, metrics: float | None = None, **_: Any) -> None:
        """
        Updates the scheduler for the network

        Parameters
        ----------
        metrics : float, default = None
            Loss metric to update ReduceLROnPlateau
        """
        learning_rate: list[float]
        new_learning_rate: list[float]
        assert isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau)

        if metrics is None:
            return

        try:
            learning_rate = self.scheduler.get_last_lr()
            self.scheduler.step(metrics)
            new_learning_rate = self.scheduler.get_last_lr()

            if learning_rate[-1] != new_learning_rate[-1]:
                print(f'Learning rate update: {new_learning_rate[-1]:.3e}')
        except AttributeError:
            self.scheduler.step(metrics)

    def _update_epoch(self) -> None:
        """
        Updates network epoch
        """
        self._epoch += 1

    @staticmethod
    def _save_predictions(path: str | None, data: dict[str, ndarray]) -> None:
        """
        Saves network predictions to pickle file if path is provided

        Parameters
        ----------
        path : str
            Path to save network predictions
        data : dict[str, (N,...) ndarray]
            Network predictions to save for dataset of size N
        """
        if path:
            with open(path, 'wb') as file:
                pickle.dump(data, file)

    def train(self, train: bool) -> None:
        """
        Flips the train/eval state of the network

        Parameters
        ----------
        train : bool
            If the network should be in the train state
        """
        self._train_state = train

        if self._train_state:
            self.net.train()
        else:
            self.net.eval()

    def training(self, epochs: int, loaders: tuple[DataLoader, DataLoader]) -> None:
        """
        Trains & validates the network for each epoch

        Parameters
        ----------
        epochs : int
            Number of epochs to train the network up to
        loaders : tuple[DataLoader, DataLoader]
            Train and validation data loaders
        """
        t_initial: float
        final_loss: float

        # Train for each epoch
        for i in range(self._epoch, epochs):
            t_initial = time()

            # Train network
            self.train(True)
            self.losses[0].append(self._train_val(loaders[0]))

            # Validate network
            self.train(False)
            self.losses[1].append(self._train_val(loaders[1]))
            self._update_scheduler(metrics=self.losses[1][-1])

            # Save training progress
            self._update_epoch()
            self.save()

            if self._verbose in ('full', 'epoch'):
                print(f'Epoch [{self._epoch}/{epochs}]\t'
                      f'Training loss: {self.losses[0][-1]:.3e}\t'
                      f'Validation loss: {self.losses[1][-1]:.3e}\t'
                      f'Time: {time() - t_initial:.1f}')
            elif self._verbose == 'progress':
                progress_bar(
                    i,
                    epochs,
                    text=f'Epoch [{self._epoch}/{epochs}]\t'
                         f'Training loss: {self.losses[0][-1]:.3e}\t'
                         f'Validation loss: {self.losses[1][-1]:.3e}\t'
                         f'Time: {time() - t_initial:.1f}',
                )

        self.train(False)
        final_loss = self._train_val(loaders[1])
        print(f'\nFinal validation loss: {final_loss:.3e}')

    def save(self) -> None:
        """
        Saves the network to the given path
        """
        if self.save_path:
            self.net.checkpoints = []
            torch.save(self, self.save_path)

    def predict(
            self,
            loader: DataLoader,
            path: str | None = None,
            **kwargs: Any) -> dict[str, ndarray]:
        """
        Generates predictions for the network and can save to a file

        Parameters
        ----------
        loader : DataLoader
            Data loader to generate predictions for
        path : str, default = None
            Path as CSV file to save the predictions if they should be saved

        **kwargs
            Optional keyword arguments to pass to batch_predict

        Returns
        -------
        dict[str, (N,...) ndarray]
            Prediction IDs, target values and predicted values for dataset of size N
        """
        initial_time: float = time()
        key: str
        ids: tuple[str, ...] | ndarray | Tensor
        data: list[list[ndarray]] = []
        data_: dict[str, ndarray]
        value: ndarray
        target: Tensor
        in_data: Tensor
        low_dim: Tensor
        high_dim: Tensor
        transform: BaseTransform | None
        self.train(False)

        # Generate predictions
        with torch.no_grad(), torch.autocast(
                enabled=self._half,
                device_type=self._device.type,
                dtype=torch.float32):
            for i, (ids, low_dim, high_dim, *_) in enumerate(loader):
                in_data, target = self._data_loader_translation(low_dim, high_dim)
                data.append([
                    ids.numpy() if isinstance(ids, Tensor) else np.array(ids),
                    target.numpy(),
                    *self.batch_predict(in_data.to(self._device), **kwargs),
                ])

                if self._verbose == 'full':
                    progress_bar(i, len(loader))

        data_ = {
            key: np.concatenate(value) if transform is None
            else transform(np.concatenate(value), back=True)
            for (key, transform), value in zip(self.header.items(), zip(*data))
        }
        print(f'Prediction time: {time() - initial_time:.3e} s')
        self._save_predictions(path, data_)
        return data_

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
        return (self.net(data).detach().cpu().numpy(),)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        r"""Move and/or cast the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self
        """
        self.net = self.net.to(*args, **kwargs)
        self._param_device(self.optimiser.state, *args, **kwargs)
        self._device, *_ = torch._C._nn._parse_to(*args, **kwargs)  # pylint: disable=protected-access
        return self


def load_net(num: int, states_dir: str, net_name: str) -> BaseNetwork:
    """
    Loads a network from file

    Parameters
    ----------
    num : int
        File number of the saved state
    states_dir : str
        Directory to the save files
    net_name : str
        Name of the network

    Returns
    -------
    BaseNetwork
        Saved network object
    """
    path: str = save_name(num, states_dir, net_name)
    return torch.load(path, map_location='cpu')
