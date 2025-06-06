"""
Base network class to base other networks off
"""
import os
import pickle
import logging as log
from time import time
from warnings import warn
from typing import Any, Self, Union, Iterable

import torch
import numpy as np
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from numpy import ndarray

from netloader.network import Network
from netloader.transforms import BaseTransform
from netloader.utils.utils import save_name, progress_bar

Param = dict[Any, Union[torch.Tensor, 'Param']]


class BaseNetwork:
    """
    Base network class that other types of networks build from

    Attributes
    ----------
    net : Module | Network
        Neural network
    save_path : str, default = ''
        Path to the network save file
    description : str, default = ''
        Description of the network
    losses : tuple[list[float], list[float]], default = ([], [])
        Network training and validation losses
    transforms : dict[str, BaseTransform | None], default = {...: None, ...}
        Keys for the output data from predict and corresponding transforms
    idxs: (N) ndarray, default = None
        Data indices for random training & validation datasets
    optimiser : Optimizer | None, default = None
        Network optimiser, if learning rate is provided AdamW will be used
    scheduler : LRScheduler, default = None
        Optimiser scheduler, if learning rate is provided ReduceLROnPlateau will be used

    Methods
    -------
    set_optimiser(parameters, **kwargs) -> Optimizer
        Sets the optimiser for the network
    set_scheduler(optimiser, **kwargs) -> LRScheduler
        Sets the scheduler for the network
    get_device() -> device
        Gets the device of the network
    get_epochs() -> int
        Returns the number of epochs the network has been trained for
    train(train)
        Flips the train/eval state of the network
    training(epoch, training)
        Trains & validates the network for each epoch
    save()
        If save_num is provided, saves the network to the states directory
    predict(loader, path=None, **kwargs) -> dict[str, (N,...) ndarray]
        Generates predictions for a dataset and can save to a file
    batch_predict(data) -> tuple[(N,...) ndarray]
        Generates predictions for the given data batch
    to(*args, **kwargs) -> Self
        Move and/or cast the parameters and buffers
    extra_repr() -> str
        Displays layer parameters when printing the architecture
    """
    def __init__(
            self,
            save_num: int | str,
            states_dir: str,
            net: nn.Module | Network,
            overwrite: bool = False,
            mix_precision: bool = False,
            learning_rate: float = 1e-3,
            description: str = '',
            verbose: str = 'epoch',
            transform: BaseTransform | None = None,
            in_transform: BaseTransform | None = None,
            optimiser_kwargs: dict[str, Any] | None = None,
            scheduler_kwargs: dict[str, Any] | None = None) -> None:
        """
        Parameters
        ----------
        save_num : int | str
            File number or name to save the network
        states_dir : str
            Directory to save the network
        net : Module | Network
            Network to predict low-dimensional data
        overwrite : bool, default = False
            If saving can overwrite an existing save file, if True and file with the same name
            exists, an error will be raised
        mix_precision: bool, default = False
            If mixed precision should be used
        learning_rate : float, default = 1e-3
            Optimiser initial learning rate
        description : str, default = ''
            Description of the network
        verbose : {'epoch', 'full', 'progress', None}
            If details about each epoch should be printed ('epoch'), details about epoch and epoch
            progress (full), just total progress ('progress'), or nothing (None)
        transform : BaseTransform, default = None
            Transformation of the network's output
        in_transform : BaseTransform, default = None
            Transformation for the input data
        optimiser_kwargs : dict[str, Any] | None, default = None
            Optional keyword arguments to pass to set_optimiser
        scheduler_kwargs : dict[str, Any] | None, default = None
            Optional keyword arguments to pass to set_scheduler
        """
        self._train_state: bool = True
        self._half: bool = mix_precision
        self._epoch: int = 0
        self._verbose: str = verbose
        self._device: torch.device = torch.device('cpu')
        self.save_path: str = ''
        self.description: str = description
        self.losses: tuple[list[float], list[float]] = ([], [])
        self.transforms: dict[str, BaseTransform | None] = {
            'ids': None,
            'inputs': in_transform,
            'targets': transform,
            'preds': transform,
        }
        self.idxs: ndarray | None = None
        self.optimiser: optim.Optimizer
        self.scheduler: optim.lr_scheduler.LRScheduler
        self.net: nn.Module | Network = net

        if not isinstance(self.net, Network) and not hasattr(self.net, 'net'):
            self.net.net = nn.ModuleList(self.net._modules.values())
        elif not isinstance(self.net, Network) and not isinstance(self.net.net, nn.ModuleList):
            raise NameError('net requires an indexable module list as attribute net, but attribute '
                            'net already exists')

        if not hasattr(self.net, 'name'):
            self.net.name = type(self.net).__name__

        if save_num:
            self.save_path = save_name(save_num, states_dir, self.net.name)

            if os.path.exists(self.save_path) and overwrite:
                log.getLogger(__name__).warning(f'{self.save_path} already exists and will be '
                                                f'overwritten if training continues')
            elif os.path.exists(self.save_path):
                raise FileExistsError(f'{self.save_path} already exists and overwrite is False')

        self.optimiser = self.set_optimiser(
            self.net.parameters(),
            lr=learning_rate,
            **optimiser_kwargs or {},
        )
        self.scheduler = self.set_scheduler(self.optimiser, **scheduler_kwargs or {})

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.load_state_dict(
                {'factor': 0.5, 'min_lr': learning_rate * 1e-3} | (scheduler_kwargs or {}),
            )

        # Adds all network classes to list of safe PyTorch classes when loading saved networks
        torch.serialization.add_safe_globals([self.__class__])

    def __repr__(self) -> str:
        """
        Returns a string representation of the network

        Returns
        -------
        str
            String representation of the network
        """
        return (f'Architecture: {self.__class__.__name__}\n'
                f'Description: {self.description}\n'
                f'Network: {self.net.name}\n'
                f'Epoch: {self._epoch}\n'
                f'Optimiser: {self.optimiser.__class__.__name__}\n'
                f'Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else None}\n'
                f'Args: ({self.extra_repr()})')

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the network for pickling

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the network
        """
        self._param_device(self.optimiser.state, 'cpu')
        return {
            'half': self._half,
            'epoch': self._epoch,
            'verbose': self._verbose,
            'save_path': self.save_path,
            'description': self.description,
            'losses': self.losses,
            'transforms': self.transforms,
            'idxs': None if self.idxs is None else self.idxs.tolist(),
            'optimiser': self.optimiser.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'net': self.net,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the network for pickling

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the network
        """
        self._train_state = True
        self._half = state['half']
        self._epoch = state['epoch']
        self._verbose = state['verbose']
        self._device = torch.device('cpu')
        self.save_path = state['save_path']
        self.description = state['description']
        self.losses = state['losses']
        self.transforms = state['transforms'] if 'transforms' in state else state['header']
        self.idxs = state['idxs'] if state['idxs'] is None else np.array(state['idxs'])
        self.net = state['net']
        self.optimiser = self.set_optimiser(self.net.parameters())
        self.scheduler = self.set_scheduler(self.optimiser)

        if 'header' in state:
            warn(
                'header attribute of BaseNetwork is deprecated, please resave the network '
                'with the new attribute name using net.save()',
                DeprecationWarning,
                stacklevel=2,
            )
            self.transforms['inputs'] = state['in_transform']

        if isinstance(state['optimiser'], dict):
            self.optimiser.load_state_dict(state['optimiser'])
            self.scheduler.load_state_dict(state['scheduler'])
        else:
            warn(
                'Optimiser & scheduler is saved in old non-weights safe format and is'
                ' deprecated, please resave the network in the new format using net.save()',
                DeprecationWarning,
                stacklevel=2,
            )
            self.optimiser = state['optimiser']
            self.scheduler = state['scheduler']

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
        t_initial: float
        loss: float = 0
        low_dim: Tensor
        high_dim: Tensor

        with torch.set_grad_enabled(self._train_state), torch.autocast(
                enabled=self._half,
                device_type=self._device.type,
                dtype=torch.bfloat16):
            for i, (_, low_dim, high_dim, *_) in enumerate(loader):
                t_initial = time()
                low_dim = low_dim.to(self._device)
                high_dim = high_dim.to(self._device)
                loss += self._loss(*self._data_loader_translation(low_dim, high_dim))
                self._update_scheduler(epoch=self._epoch + (i + 1) / len(loader))

                if self._verbose == 'full':
                    progress_bar(
                        i,
                        len(loader),
                        text=f'Average loss: {loss / (i + 1):.2e}\tTime: {time() - t_initial:.1f}',
                    )

        return loss / len(loader)

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
        raise NotImplementedError

    def _param_device(self, params: Param, *args: Any, **kwargs: Any) -> None:
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
        # pylint: disable=protected-access
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
        if not path:
            return

        with open(path + '' if '.pkl' in path else '.pkl', 'wb') as file:
            pickle.dump(data, file)  # type: ignore

    @staticmethod
    def set_optimiser(
            parameters: list[dict[str, Iterable[nn.Parameter]]] | Iterable[nn.Parameter],
            **kwargs: Any) -> optim.Optimizer:
        """
        Sets the optimiser for the network, by default it is AdamW

        Parameters
        ----------
        parameters : list[dict[str, Iterable[nn.Parameter]]] | Iterable[nn.Parameter]
            Network parameters to optimise

        **kwargs
            Optional keyword arguments to pass to the optimiser

        Returns
        -------
        optim.Optimizer
            Network optimiser
        """
        return optim.AdamW(parameters, **kwargs)

    @staticmethod
    def set_scheduler(optimiser: optim.Optimizer, **kwargs: Any) -> optim.lr_scheduler.LRScheduler:
        """
        Sets the scheduler for the network, by default it is ReduceLROnPlateau

        Parameters
        ----------
        optimiser : optim.Optimizer
            Network optimiser
        **kwargs
            Optional keyword arguments to pass to the scheduler

        Returns
        -------
        optim.lr_scheduler.LRScheduler
            Optimiser scheduler
        """
        return optim.lr_scheduler.ReduceLROnPlateau(optimiser, **kwargs)

    def get_device(self) -> torch.device:
        """
        Gets the device of the network

        Returns
        -------
        device
            Device of the network
        """
        return self._device

    def get_epochs(self) -> int:
        """
        Returns the number of epochs the network has been trained for

        Returns
        -------
        int
            Number of epochs
        """
        return self._epoch

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
            torch.save(self, self.save_path)
            self.to(self._device)

    def predict(
            self,
            loader: DataLoader,
            inputs: bool = False,
            path: str | None = None,
            **kwargs: Any) -> dict[str, ndarray]:
        """
        Generates predictions for the network and can save to a file

        Parameters
        ----------
        loader : DataLoader
            Data loader to generate predictions for
        inputs : bool, default = False,
            If the input data should be returned and saved
        path : str, default = None
            Path as CSV file to save the predictions if they should be saved

        **kwargs
            Optional keyword arguments to pass to batch_predict

        Returns
        -------
        dict[str, (N,...) ndarray]
            Prediction IDs, optional inputs, target values, and predicted values for dataset of
            size N
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

        if 'input_' in kwargs:
            warn(
                'input_ keyword argument is deprecated, please use inputs instead',
                DeprecationWarning,
                stacklevel=2,
            )
            inputs = kwargs.pop('input_')

        # Generate predictions
        with torch.no_grad(), torch.autocast(
                enabled=self._half,
                device_type=self._device.type,
                dtype=torch.float32):
            for i, (ids, low_dim, high_dim, *_) in enumerate(loader):
                in_data, target = self._data_loader_translation(low_dim, high_dim)
                data.append([
                    ids.numpy() if isinstance(ids, Tensor) else np.array(ids),
                    *([in_data.numpy()] if inputs else []),
                    target.numpy(),
                    *self.batch_predict(in_data.to(self._device), **kwargs),
                ])

                if self._verbose == 'full':
                    progress_bar(i, len(loader))

        # Transforms all data and saves it to a dictionary
        transforms = {key: value for key, value in self.transforms.items()
                      if inputs or key != 'inputs'}
        data_ = {
            # Concatenate if there is no transform
            key: np.concat(value) if transform is None
            # If prediction is a tuple, treat second entry as the uncertainty and apply transform
            else transform(
                np.concat(np.array(value)[:, 0]),
                back=True,
                uncertainty=np.concat(np.array(value)[:, 1]),
            )
            if isinstance(value[0], tuple)
            # Else apply transformation
            else transform(np.concat(value), back=True)
            for (key, transform), value in zip(transforms.items(), zip(*data))
        }

        if self._verbose is not None:
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

    def extra_repr(self) -> str:
        """
        Additional representation of the architecture

        Returns
        -------
        str
            Architecture specific representation
        """
        return ''


def load_net(
        num: int | str,
        states_dir: str,
        net_name: str,
        weights_only: bool = True) -> BaseNetwork:
    """
    Loads a network from file

    Parameters
    ----------
    num : int | str
        File number or name of the saved state
    states_dir : str
        Directory to the save files
    net_name : str
        Name of the network
    weights_only : bool, default = True
        If PyTorch should only load tensors, primitive types, dictionaries & types added to the
        torch.serialization.add_safe_globals()

    Returns
    -------
    BaseNetwork
        Saved network object
    """
    return torch.load(
        save_name(num, states_dir, net_name),
        map_location='cpu',
        weights_only=weights_only,
    )
