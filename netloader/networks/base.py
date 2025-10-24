"""
Base network class to base other networks off
"""
import os
import logging as log
from time import time
from warnings import warn
from typing import Any, Self, Generic, Literal, cast

import torch
import numpy as np
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from numpy import ndarray

import netloader
import netloader.data
from netloader import utils
from netloader.transforms import BaseTransform
from netloader.networks.utils import UtilityMixin
from netloader.network import Network, CompatibleNetwork
from netloader.data import Data, DataList, data_collation
from netloader.utils.types import (
    Param,
    TensorLike,
    NDArrayLike,
    TensorListLike,
    NDArrayListLike,
    LossCT,
    TensorLossCT,
)


class BaseNetwork(UtilityMixin, Generic[LossCT, TensorLossCT]):
    """
    Base network class that other types of networks build from

    Attributes
    ----------
    save_path : str
        Path to the network save file
    description : str
        Description of the network
    version : str
        Version of the network when it was created or re-saved
    losses : tuple[list[LossCT], list[LossCT]]
        Network training and validation losses as a float or dictionary of losses for each loss
        function
    transforms : dict[str, list[BaseTransform] | BaseTransform | None]
        Keys for the output data from predict and corresponding transforms
    idxs: ndarray | None
        Training data indices with shape (N) and type int, where N is the number of elements in the
        training dataset
    net : nn.Module | Network | CompatibleNetwork
        Neural network
    optimiser : optim.Optimizer
        Network optimiser
    scheduler : optim.lr_scheduler.LRScheduler
        Optimiser scheduler

    Methods
    -------
    extra_repr() -> str
        Additional representation of the architecture
    get_device() -> torch.device
        Gets the device of the network
    get_epochs() -> int
        Returns the number of epochs the network has been trained for
    get_hyperparams() -> dict[str, Any]
        Returns the hyperparameters of the network
    train(train)
        Flips the train/eval state of the network
    training(epoch, loaders)
        Trains & validates the network for each epoch
    save()
        Saves the network to the given path
    predict(loader, *, inputs=False, path='', **kwargs) -> dict[str, NDArrayLike]
        Generates predictions for a dataset and can save to a file
    batch_predict(data) -> tuple[NDArrayLike | None, ...]
        Generates predictions for the given data batch
    to(*args, **kwargs) -> Self
        Move and/or cast the parameters and buffers
    """
    def __init__(
            self,
            save_num: int | str,
            states_dir: str,
            net: nn.Module | Network,
            *,
            overwrite: bool = False,
            mix_precision: bool = False,
            save_freq: int = 1,
            learning_rate: float = 1e-3,
            description: str = '',
            verbose: Literal['epoch', 'full', 'plot', 'progress', None] = 'epoch',
            transform: list[BaseTransform] | BaseTransform | None = None,
            in_transform: list[BaseTransform] | BaseTransform | None = None,
            optimiser_kwargs: dict[str, Any] | None = None,
            scheduler_kwargs: dict[str, Any] | None = None) -> None:
        """
        Parameters
        ----------
        save_num : int | str
            File number or name to save the network
        states_dir : str
            Directory to save the network
        net : nn.Module | Network
            Network to predict low-dimensional data
        overwrite : bool, default = False
            If saving can overwrite an existing save file, if True and file with the same name
            exists, an error will be raised
        mix_precision: bool, default = False
            If mixed precision should be used
        save_freq : int, default = 1
            Frequency of epochs to save the network
        learning_rate : float, default = 1e-3
            Optimiser initial learning rate
        description : str, default = ''
            Description of the network
        verbose : {'epoch', 'full', 'plot', 'progress', None}
            If details about each epoch should be printed ('epoch'), details about epoch and epoch
            progress (full), details about epoch and an ASCII plot of the loss progress ('plot'),
            just total progress ('progress'), or nothing (None)
        transform : list[BaseTransform] | BaseTransform | None, default = None
            Transformation(s) of the network's output(s)
        in_transform : list[BaseTransform] | BaseTransform | None, default = None
            Transformation(s) for the network's input(s)
        optimiser_kwargs : dict[str, Any] | None, default = None
            Optional keyword arguments to pass to set_optimiser
        scheduler_kwargs : dict[str, Any] | None, default = None
            Optional keyword arguments to pass to set_scheduler
        """
        self._train_state: bool = True
        self._plot_active: bool = False
        self._half: bool = mix_precision
        self._epoch: int = 0
        self._save_freq: int = save_freq
        self._verbose: Literal['epoch', 'full', 'plot', 'progress', None] = verbose
        self._optimiser_kwargs: dict[str, Any] = optimiser_kwargs or {}
        self._scheduler_kwargs: dict[str, Any] = scheduler_kwargs or {}
        self._logger: log.Logger = log.getLogger(__name__)
        self._device: torch.device = torch.device('cpu')
        self.save_path: str = ''
        self.description: str = description
        self.version: str = netloader.__version__
        self.losses: tuple[list[LossCT], list[LossCT]] = ([], [])
        self.transforms: dict[str, list[BaseTransform] | BaseTransform | None] = {
            'ids': None,
            'inputs': in_transform,
            'targets': transform,
            'preds': transform,
        }
        self.idxs: ndarray | None = None
        self.optimiser: optim.Optimizer
        self.scheduler: optim.lr_scheduler.LRScheduler
        self.net: Network | CompatibleNetwork = net if isinstance(net, Network) else \
            CompatibleNetwork(net)

        if save_num:
            self.save_path = utils.save_name(save_num, states_dir, self.net.name)

            if os.path.exists(self.save_path) and overwrite:
                self._logger.warning(f'{self.save_path} already exists and will be overwritten if '
                                     f'training continues')
            elif os.path.exists(self.save_path):
                raise FileExistsError(f'{self.save_path} already exists and overwrite is False')

        self.optimiser = self.set_optimiser(
            self.net.parameters(),
            lr=learning_rate,
            **self._optimiser_kwargs,
        )
        self.scheduler = self.set_scheduler(self.optimiser, **self._scheduler_kwargs)

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self._scheduler_kwargs = ({'factor': 0.5, 'min_lr': learning_rate * 1e-3}
                                      | self._scheduler_kwargs)
            self.scheduler.load_state_dict(self._scheduler_kwargs)

        # Adds all network classes to list of safe PyTorch classes when loading saved networks
        torch.serialization.add_safe_globals([self.__class__])

    def __repr__(self) -> str:
        """
        Returns a string representation of the network.

        Returns
        -------
        str
            String representation of the network
        """
        return (f'Architecture: {self.__class__.__name__}\n'
                f'Description: {self.description}\n'
                f'Version: {self.version}\n'
                f'Network: {self.net.name}\n'
                f'Epoch: {self._epoch}\n'
                f'Optimiser: {self.optimiser.__class__.__name__}\n'
                f'Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else None}\n'
                f'Args: ({self.extra_repr()})')

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the network for pickling.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the network
        """
        return {
            'half': self._half,
            'epoch': self._epoch,
            'save_freq': self._save_freq,
            'verbose': self._verbose,
            'save_path': self.save_path,
            'description': self.description,
            'version': netloader.__version__,
            'losses': self.losses,
            'optimiser_kwargs': self._optimiser_kwargs,
            'scheduler_kwargs': self._scheduler_kwargs,
            'transforms': self.transforms,
            'idxs': None if self.idxs is None else self.idxs.tolist(),
            'optimiser': self.optimiser.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'net': self.net,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the network for pickling.

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the network
        """
        self._train_state = True
        self._plot_active = False

        for key, value in list(state.items()):
            if key[0] == '_':
                warn(
                    'BaseNetwork is saved in old format and is deprecated, please resave '
                    'using BaseNetwork.save()',
                    DeprecationWarning,
                    stacklevel=2,
                )
                state[key.replace('_', '', 1)] = value

        self._half = state['half']
        self._epoch = state['epoch']
        self._save_freq = state.get('save_freq', 1)
        self.version = state.get('version', '<3.7.1')
        self._verbose = state['verbose']
        self._device = torch.device('cpu')
        self.save_path = state['save_path']
        self.description = state['description']
        self.losses = state['losses']
        self._optimiser_kwargs = state.get('optimiser_kwargs', {})
        self._scheduler_kwargs = state.get('scheduler_kwargs', {})
        self.transforms = state['transforms'] if 'transforms' in state else state['header']
        self.idxs = state['idxs'] if state['idxs'] is None else np.array(state['idxs'])
        self.net = state['net']

        if 'header' in state:
            warn(
                'header attribute of BaseNetwork is deprecated, please resave the network '
                'with the new attribute name using net.save()',
                DeprecationWarning,
                stacklevel=2,
            )
            self.transforms['inputs'] = state['in_transform']

        if isinstance(state['optimiser'], dict):
            self.optimiser = self.set_optimiser(
                self.net.parameters(),
                **state.get('optimiser_kwargs', {}),
            )
            self.scheduler = self.set_scheduler(
                self.optimiser,
                **state.get('scheduler_kwargs', {}),
            )
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

    def _batch_print(
            self,
            i: int,
            batch_time: float,
            loader: DataLoader[Any],
            loss: LossCT) -> None:
        """
        Print function during each batch of training.

        Parameters
        ----------
        i : int
            Batch number
        batch_time : float
            Time taken for the batch
        loader: DataLoader[Any]
            Data loader that the batch came from
        loss : LossCT
            Loss for the batch or a dictionary of losses
        """
        if self._verbose == 'full':
            utils.progress_bar(
                i,
                len(loader),
                text=f'Average loss: '
                     f'{(sum(loss.values()) if isinstance(loss, dict) else loss) / (i + 1):.2e}\t'
                     f'Time: {batch_time:.1f}',
            )

    def _epoch_print(
            self,
            i: int,
            epochs: int,
            epoch_time: float) -> None:
        """
        Print function at the end of each epoch.

        Parameters
        ----------
        i : int
            Current epoch number
        epochs : int
            Total number of epochs
        epoch_time : float
            Time taken for the epoch
        """
        text: str
        losses: tuple[list[float], list[float]] = (
            [loss['total'] if isinstance(loss, dict) else loss for loss in self.losses[0]],
            [loss['total'] if isinstance(loss, dict) else loss for loss in self.losses[1]],
        )
        loss: LossCT

        text = f'Epoch [{self._epoch}/{epochs}]\t' \
               f'Training loss: {losses[0][-1]:.3e}\t' \
               f'Validation loss: {losses[1][-1]:.3e}\t' \
               f'Time: {epoch_time:.1f}'

        if (self._verbose in {'full', 'epoch'} or
                (len(losses[0]) == 1 and self._verbose == 'plot')):
            print(text)
        elif self._verbose == 'progress':
            utils.progress_bar(i, epochs, text=text)
        elif self._verbose == 'plot':
            utils.ascii_plot(
                losses[0],
                clear=self._plot_active,
                text=text,
                data2=losses[1],
            )
            self._plot_active = True

    def _predict_print(self, i: int, predict_time: float, loader: DataLoader[Any]) -> None:
        """
        Print function during each batch of prediction.

        Parameters
        ----------
        i : int
            Batch number
        predict_time : float
            Time taken for predicting
        loader: DataLoader[Any]
            Data loader that is being used for predicting
        """
        if self._verbose == 'full':
            utils.progress_bar(i, len(loader))

        if i == len(loader) - 1 and self._verbose is not None:
            print(f'Prediction time: {predict_time:.3e} s')


    def _train_val(self, loader: DataLoader[Any]) -> LossCT:
        """
        Trains the network for one epoch.

        Parameters
        ----------
        loader : DataLoader
            PyTorch DataLoader that contains data to train

        Returns
        -------
        LossCT
            Average loss value or dictionary of average loss values for the epoch
        """
        i: int
        value: float
        t_initial: float
        key: str
        loss: LossCT | None = None
        low_dim: list[Tensor] | list[Data[Tensor]] | list[DataList[Tensor | Data[Tensor]]]
        high_dim: list[Tensor] | list[Data[Tensor]] | list[DataList[Tensor | Data[Tensor]]]
        batch_loss: LossCT
        target: TensorListLike
        in_data: TensorListLike

        with torch.set_grad_enabled(self._train_state), torch.autocast(
                enabled=self._half,
                device_type=self._device.type,
                dtype=torch.bfloat16):
            for i, (_, low_dim, high_dim, *_) in enumerate(loader):
                t_initial = time()
                in_data, target = cast(
                    tuple[TensorListLike, TensorListLike],
                    self._data_loader_translation(
                        data_collation(low_dim, data_field=False).to(self._device),
                        data_collation(high_dim, data_field=False).to(self._device),
                    ),
                )
                batch_loss = self._loss(in_data, target)

                if isinstance(batch_loss, dict) and loss:
                    for key, value in batch_loss.items():
                        loss[key] += value

                    loss['total'] += batch_loss['total'] if 'total' in batch_loss else \
                        sum(batch_loss.values())
                elif isinstance(batch_loss, float) and loss:
                    loss += batch_loss
                else:
                    loss = batch_loss

                if self._train_state:
                    self._update_scheduler(epoch=self._epoch + (i + 1) / len(loader))

                self._batch_print(i, time() - t_initial, loader, batch_loss)

        assert loss

        if isinstance(loss, dict):
            return {key: value / len(loader) for key, value in loss.items()}
        return loss / len(loader)

    def _loss_func(self, in_data: TensorListLike, target: TensorListLike) -> TensorLossCT:
        """
        Empty method for child classes to base their loss functions on.

        Parameters
        ----------
        in_data : TensorListLike
            Input data of shape (N,...) and type float, where N is the number of elements
        target : TensorListLike
            Target data of shape (N,...) and type float

        Returns
        -------
        TensorLossCT
            Loss of shape (1) and type float or dictionary of losses of shape (1) and type float
        """
        raise DeprecationWarning

    def _loss_tensor(self, in_data: TensorListLike, target: TensorListLike) -> TensorLossCT:
        """
        Empty method for child classes to base their loss functions on.

        Parameters
        ----------
        in_data : TensorListLike
            Input data of shape (N,...) and type float, where N is the number of elements
        target : TensorListLike
            Target data of shape (N,...) and type float

        Returns
        -------
        TensorLossCT
            Loss of shape (1) and type float or dictionary of losses of shape (1) and type float
        """
        raise NotImplementedError

    def _loss(self, in_data: TensorListLike, target: TensorListLike) -> LossCT:
        """
        Returns the loss as a float & updates network weights if training.

        Parameters
        ----------
        in_data : TensorListLike
            Input data of shape (N,...) and type float, where N is the number of elements
        target : TensorListLike
            Target data of shape (N,...) and type float

        Returns
        -------
        LossCT
            Loss or dictionary of losses which can be summed to get the total loss
        """
        key: str
        value: Tensor
        loss: TensorLossCT

        try:
            loss = self._loss_func(in_data, target)
            warn(
                '_loss_func is deprecated, please use _loss_tensor instead',
                DeprecationWarning,
                stacklevel=2,
            )
        except DeprecationWarning:
            loss = self._loss_tensor(in_data, target)

        self._update(
            loss if isinstance(loss, Tensor) else
            loss['total'] if 'total' in loss else
            torch.sum(torch.cat(list(loss.values()))),
        )

        if isinstance(loss, dict):
            return {key: value.item() for key, value in loss.items()}  # type: ignore[return-value]
        return loss.item()  # type: ignore[return-value]

    def _param_device(self, params: dict[Tensor, Param] | Param, *args: Any, **kwargs: Any) -> None:
        """
        Sends parameters to the device, such as the parameters in the optimiser.

        Parameters
        ----------
        params : dict[Tensor, Param] | Param
            Parameters to move/cast
        *args, **kwargs
            Arguments to pass to torch.Tensor.to
        """
        param: Tensor | Param

        for param in params.values():
            if isinstance(param, dict):
                self._param_device(param, *args, **kwargs)
            elif isinstance(param, Tensor):
                param.data = param.data.to(*args, **kwargs)

                # pylint: disable=protected-access
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(*args, **kwargs)
                # pylint: enable=protected-access

    def _update(self, loss: Tensor) -> None:
        """
        Updates the network using backpropagation.

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
        Updates the scheduler for the network.

        Parameters
        ----------
        metrics : float | None, default = None
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
        Updates network epoch.
        """
        self._epoch += 1

    def extra_repr(self) -> str:
        """
        Additional representation of the architecture.

        Returns
        -------
        str
            Architecture specific representation
        """
        return ''

    def get_device(self) -> torch.device:
        """
        Gets the device of the network.

        Returns
        -------
        torch.device
            Device of the network
        """
        return self._device

    def get_epochs(self) -> int:
        """
        Returns the number of epochs the network has been trained for.

        Returns
        -------
        int
            Number of epochs
        """
        return self._epoch

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Returns the hyperparameters of the network.

        Returns
        -------
        dict[str, Any]
            Hyperparameters of the network
        """
        return {
            'mix_precision': self._half,
            'save_freq': self._save_freq,
            'verbose': self._verbose,
            'optimiser_kwargs': self._optimiser_kwargs,
            'scheduler_kwargs': self._scheduler_kwargs,
            'architecture_name': self.__class__.__name__,
            'description': self.description,
            'net_name': self.net.name,
            'version': self.version,
            'optimiser': self.optimiser.__class__.__name__,
            'scheduler': self.scheduler.__class__.__name__,
        }

    def train(self, train: bool) -> None:
        """
        Flips the train/eval state of the network.

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

    def training(self, epochs: int, loaders: tuple[DataLoader[Any], DataLoader[Any]]) -> None:
        """
        Trains & validates the network for each epoch.

        Parameters
        ----------
        epochs : int
            Number of epochs to train the network up to
        loaders : tuple[DataLoader[Any], DataLoader[Any]]
            Train and validation data loaders
        """
        i: int
        t_initial: float
        loss: LossCT

        # Train for each epoch
        for i in range(self._epoch, epochs):
            t_initial = time()

            # Train network
            self.train(True)
            self.losses[0].append(self._train_val(loaders[0]))

            # Validate network
            self.train(False)
            self.losses[1].append(self._train_val(loaders[1]))
            self._update_scheduler(
                metrics=self.losses[1][-1]['total'] if isinstance(self.losses[1][-1], dict) else \
                    self.losses[1][-1],
            )

            # Save training progress
            self._update_epoch()
            self.save()
            self._epoch_print(i, epochs, time() - t_initial)

        self.train(False)
        self._plot_active = False
        loss = self._train_val(loaders[1])
        print(f"\nFinal validation loss: {loss if isinstance(loss, float) else loss['total']:.3e}")

    def save(self) -> None:
        """
        Saves the network to the given path.
        """
        if self.save_path and (self._epoch - 1) % self._save_freq == 0:
            try:
                torch.save(self, self.save_path)
            except KeyboardInterrupt:
                print('Program interrupted, finishing network save...')
                torch.save(self, self.save_path)

    def predict(
            self,
            loader: DataLoader[Any],
            *,
            inputs: bool = False,
            path: str = '',
            **kwargs: Any) -> dict[str, NDArrayLike]:
        """
        Generates predictions for the network and can save to a file.

        Parameters
        ----------
        loader : DataLoader[Any]
            Data loader to generate predictions for
        inputs : bool, default = False
            If the input data should be returned and saved
        path : str, default = ''
            Path as pkl file to save the predictions if they should be saved
        **kwargs
            Optional keyword arguments to pass to batch_predict

        Returns
        -------
        dict[str, NDArrayLike]
            Prediction IDs, optional inputs, target values, and predicted values of shape (N,...)
            and type float for dataset of size N
        """
        t_initial: float = time()
        key: str
        ids: tuple[str, ...] | ndarray | Tensor
        low_dim: list[Tensor] | list[Data[Tensor]] | list[DataList[Tensor | Data[Tensor]]]
        high_dim: list[Tensor] | list[Data[Tensor]] | list[DataList[Tensor | Data[Tensor]]]
        data: list[list[NDArrayLike | None]] = []
        data_: dict[str, NDArrayLike] = {}
        transform: list[BaseTransform] | BaseTransform | None
        transforms: dict[str, list[BaseTransform] | BaseTransform | None] = {
            key: transform for key, transform in self.transforms.items()
            if inputs or key != 'inputs'
        }
        datum: NDArrayLike
        target: TensorLike
        in_data: TensorLike
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
                in_data, target = self._data_loader_translation(
                    data_collation(low_dim, data_field=True),
                    data_collation(high_dim, data_field=True),
                )
                data.append(cast(list[NDArrayLike | None], [
                    ids.numpy() if isinstance(ids, Tensor) else np.array(ids),
                    *([in_data.numpy()] if inputs else []),
                    target.numpy(),
                    *self.batch_predict(
                        (in_data if isinstance(in_data, Tensor) else
                        cast(DataList[Tensor], data_collation(
                            cast(list[Data] | list[DataList[Tensor]], [in_data]),
                            data_field=False,
                        ))).to(self._device),
                        **kwargs,
                    ),
                ]))
                self._predict_print(i, time() - t_initial, loader)

        # Transforms all data and saves it to a dictionary
        for (key, transform), datum_ in zip(transforms.items(), zip(*data)):
            if datum_[0] is None:
                continue

            # Concatenate values
            datum = data_collation(list(datum_), data_field=True)

            if isinstance(datum, DataList) and transform:
                data_[key] = DataList(
                    [trans(val, back=True) for val, trans in zip(
                        datum,
                        transform if isinstance(transform, list) else [transform] * len(datum),
                    )],
                )
            elif isinstance(transform, BaseTransform):
                assert not isinstance(datum, DataList)
                data_[key] = transform(datum, back=True)
            else:
                if isinstance(transform, list):
                    self._logger.warning(f'List of transforms requires corresponding data with key '
                                         f'({key}) to be a DataList, data will not be '
                                         f'untransformed')
                data_[key] = datum

        self._save_predictions(path, data_)
        return data_

    def batch_predict(self, data: TensorListLike, **_: Any) -> tuple[NDArrayListLike | None, ...]:
        """
        Generates predictions for the given data batch.

        Parameters
        ----------
        data : TensorListLike
            Data of shape (N,...) and type float to generate predictions for, where N is the batch
            size

        Returns
        -------
        tuple[NDArrayListLike | None, ...]
            Predictions of shape (N,...) and type float for the given data
        """
        return (self.net(data).detach().cpu().numpy(),)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """
        Move and/or cast the parameters and buffers.

        Parameters
        ----------
        *args, **kwargs
            Arguments to pass to torch.Tensor.to

        Returns
        -------
        Self
            The network with parameters and buffers moved/cast
        """
        self.net = self.net.to(*args, **kwargs)
        self._param_device(self.optimiser.state, *args, **kwargs)
        self._device, *_ = torch._C._nn._parse_to(*args, **kwargs)  # pylint: disable=protected-access
        return self


def load_net(num: int | str, states_dir: str, net_name: str, **kwargs: Any) -> BaseNetwork:
    """
    Loads a network from file.

    Parameters
    ----------
    num : int | str
        File number or name of the saved state
    states_dir : str
        Directory to the save files
    net_name : str
        Name of the network
    **kwargs
        Optional keyword arguments to pass to torch.load

    Returns
    -------
    BaseNetwork
        Saved network object
    """
    return torch.load(
        utils.save_name(num, states_dir, net_name),
        **{'map_location': 'cpu'} | kwargs,
    )
