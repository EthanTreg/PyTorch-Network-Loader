"""
Base architecture class to base other architectures off
"""
import os
import logging as log
from time import time
from warnings import warn
from itertools import repeat
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self, Sequence, Generic, Literal, Callable, cast, overload

import torch
import numpy as np
from numpy import ndarray
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import ParamsT
from torch._dynamo import OptimizedModule

import netloader
from netloader import utils
from netloader.transforms import BaseTransform
from netloader.architectures.utils import UtilityMixin
from netloader.data import ApplyFunc, Data, DataList, data_collation
from netloader.network import BaseNetwork, CompatibleNetwork
from netloader.utils.types import (
    TensorLike,
    NDArrayLike,
    TensorListLike,
    NDArrayListLike,
    DatasetT,
    LossCT,
    TensorLossCT,
)

if TYPE_CHECKING:
    from wandb import Run
else:
    Run = Any


class BaseArchitecture(UtilityMixin, ABC, Generic[LossCT, TensorLossCT]):
    # pylint: disable=line-too-long
    """
    Base architecture class that other types of architectures build from

    Attributes
    ----------
    description : str
        Description of the architecture
    version : str
        Version of the architecture when it was created or re-saved
    losses : tuple[list[LossCT], list[LossCT]]
        Architecture training and validation losses as a float or dictionary of losses for each loss
        function
    transforms : dict[str, list[BaseTransform] | BaseTransform | None]
        Keys for the output data from predict and corresponding transforms
    idxs: ndarray | None
        Training data indices with shape (N) and type int, where N is the number of elements in the
        training dataset
    optimiser : Optimizer
        Architecture optimiser
    scheduler : LRScheduler
        Optimiser scheduler
    run : Run | None
        Weights & Biases run object for logging and tracking experiments
    net : BaseNetwork
        Neural network
    """
    # pylint: enable=line-too-long
    _half: bool
    _train_state: bool = True
    _plot_active: bool = False
    _save_freq: int
    _epoch: int = 0
    _save_path: str = ''
    _verbose: Literal['epoch', 'full', 'plot', 'progress', None]
    _loader_states: tuple[Tensor, Tensor] | None = None
    _loss_weights: dict[str, float | Callable[[], float]]
    _optimiser_kwargs: dict[str, Any]
    _scheduler_kwargs: dict[str, Any]
    _logger: log.Logger = log.getLogger(__name__)
    _device: torch.device = torch.device('cpu')
    description: str
    version: str = netloader.__version__
    losses: tuple[list[LossCT], list[LossCT]]
    transforms: dict[str, Sequence[BaseTransform] | BaseTransform | None]
    idxs: ndarray | None = None
    optimiser: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    net: BaseNetwork
    run: Run | None = None

    def __init__(
            self,
            save_num: int | str,
            states_dir: str,
            net: nn.Module | BaseNetwork,
            *,
            overwrite: bool = False,
            mix_precision: bool = False,
            save_freq: int = 1,
            learning_rate: float | tuple[float, ...] = 1e-3,
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
            File number or name to save the architecture
        states_dir : str
            Directory to save the architecture
        net : Module | BaseNetwork
            Network to predict low-dimensional data
        overwrite : bool, Optional
            If saving can overwrite an existing save file, if False and file with the same name
            exists, an error will be raised, default = False
        mix_precision : bool, Optional
            If mixed precision should be used, default = False
        save_freq : int, Optional
            Frequency of epochs to save the architecture, default = 1
        learning_rate : float | tuple[float, ...], Optional
            Optimiser initial learning rate, default = 1e-3
        description : str, Optional
            Description of the architecture
        verbose : {'epoch', 'full', 'plot', 'progress', None}
            If details about each epoch should be printed ('epoch'), details about epoch and epoch
            progress (full), details about epoch and an ASCII plot of the loss progress ('plot'),
            just total progress ('progress'), or nothing (None)
        transform : list[BaseTransform] | BaseTransform | None, Optional
            Transformation(s) of the network's output(s)
        in_transform : list[BaseTransform] | BaseTransform | None, Optional
            Transformation(s) for the network's input(s)
        optimiser_kwargs : dict[str, Any] | None, Optional
            Optional keyword arguments to pass to init_optimiser
        scheduler_kwargs : dict[str, Any] | None, Optional
            Optional keyword arguments to pass to init_scheduler
        """
        self._half = mix_precision
        self._save_freq = save_freq
        self._verbose = verbose
        self._loss_weights = {}
        self._optimiser_kwargs = optimiser_kwargs or {}
        self._scheduler_kwargs = scheduler_kwargs or {}
        self.description = description
        self.losses = ([], [])
        self.transforms = {
            'ids': None,
            'inputs': in_transform,
            'targets': transform,
            'preds': transform,
        }
        self.net = net if isinstance(net, BaseNetwork) else CompatibleNetwork(net=net)

        if save_num:
            self._save_path = utils.save_name(save_num, states_dir, self.net.name)

            if os.path.exists(self._save_path) and overwrite:
                self._logger.warning(f'{self._save_path} already exists and will be overwritten if '
                                     f'training continues')
            elif os.path.exists(self._save_path):
                raise FileExistsError(f'{self._save_path} already exists and overwrite is False')

        if self._method_override('set_optimiser', UtilityMixin):
            warn(
                'BaseArchitecture.set_optimiser method is deprecated, please override '
                'init_optimiser instead',
                DeprecationWarning,
                stacklevel=2,
            )
            self.init_optimiser = self.set_optimiser  # type: ignore[method-assign]

        if self._method_override('set_scheduler', UtilityMixin):
            warn(
                'BaseArchitecture.set_scheduler method is deprecated, please override '
                'init_scheduler instead',
                DeprecationWarning,
                stacklevel=2,
            )
            self.init_scheduler = self.set_scheduler  # type: ignore[method-assign]

        self.optimiser = self.init_optimiser(
            self.get_param_groups(learning_rate),
            **self._optimiser_kwargs,
        )
        self.scheduler = self.init_scheduler(self.optimiser, **self._scheduler_kwargs)

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self._scheduler_kwargs = {
                'factor': 0.5,
                'min_lr': (learning_rate if isinstance(learning_rate, float) else learning_rate[0])
                          * 1e-3
            } | self._scheduler_kwargs
            self.scheduler.load_state_dict(self._scheduler_kwargs)

        # Adds all architecture classes to list of safe PyTorch classes when loading saved
        # architectures
        torch.serialization.add_safe_globals([self.__class__])

    def __repr__(self) -> str:
        """
        Returns a string representation of the architecture.

        Returns
        -------
        str
            String representation of the architecture
        """
        return (f'Architecture: {self.__class__.__name__}\n'
                f'Name: {os.path.basename(self._save_path).rsplit(".", 1)[0]}\n'
                f'Description: {self.description}\n'
                f'Version: {self.version}\n'
                f'Network: {self.net.name}\n'
                f'Epoch: {self._epoch}\n'
                f'Optimiser: {self.optimiser.__class__.__name__}\n'
                f'Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else None}\n' +
                (f'Loss Weights: {self.get_loss_weights()}\n' if self._loss_weights else '') +
                f'Args: ({self.extra_repr()})')

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the architecture for pickling.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the architecture
        """
        return {
            'half': self._half,
            'epoch': self._epoch,
            'save_freq': self._save_freq,
            'verbose': self._verbose,
            'save_path': self._save_path,
            'description': self.description,
            'version': netloader.__version__,
            'losses': self.losses,
            'loss_weights': self._loss_weights,
            'optimiser_kwargs': self._optimiser_kwargs,
            'scheduler_kwargs': self._scheduler_kwargs,
            'transforms': self.transforms,
            'rng_states': self.get_rng_states(),
            'idxs': None if self.idxs is None else self.idxs.tolist(),
            'optimiser': self.optimiser.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'net': self.net._orig_mod if isinstance(self.net, OptimizedModule) else self.net,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the architecture for pickling.

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the architecture
        """
        self._train_state = True
        self._plot_active = False

        for key, value in list(state.items()):
            if key[0] == '_':
                state[key.replace('_', '', 1)] = value

        self._half = state['half']
        self._epoch = state['epoch']
        self._save_freq = state.get('save_freq', 1)
        self._verbose = state['verbose']
        self._save_path = state['save_path']
        self._loader_states = None
        self._loss_weights = state.get('loss_weights', {})
        self._optimiser_kwargs = state.get('optimiser_kwargs', {})
        self._scheduler_kwargs = state.get('scheduler_kwargs', {})
        self._logger = log.getLogger(__name__)
        self._device = torch.device('cpu')
        self.version = state.get('version', '<3.7.1')
        self.description = state['description']
        self.losses = state['losses']
        self.transforms = state['transforms'] if 'transforms' in state else state['header']
        self.idxs = state['idxs'] if state['idxs'] is None else np.array(state['idxs'])
        self.net = state['net']
        self.run = state.get('run', None)

        if not utils.compare_versions(self.version, netloader.__version__):
            warn(
                f'Architecture version ({self.version}) is older than the current '
                f'NetLoader version ({netloader.__version__}), please resave the architecture '
                f'using BaseArchitecture.save()',
                DeprecationWarning,
                stacklevel=2,
            )

        if 'header' in state:
            warn(
                'header attribute of BaseArchitecture is deprecated, please resave the '
                'architecture with the new attribute name using BaseArchitecture.save()',
                DeprecationWarning,
                stacklevel=2,
            )
            self.transforms['inputs'] = state['in_transform']

        if self._method_override('set_optimiser', UtilityMixin):
            warn(
                'BaseArchitecture.set_optimiser method is deprecated, please override '
                'init_optimiser instead',
                DeprecationWarning,
                stacklevel=2,
            )
            self.init_optimiser = self.set_optimiser  # type: ignore[method-assign]

        if self._method_override('set_scheduler', UtilityMixin):
            warn(
                'BaseArchitecture.set_scheduler method is deprecated, please override '
                'init_scheduler instead',
                DeprecationWarning,
                stacklevel=2,
            )
            self.init_scheduler = self.set_scheduler  # type: ignore[method-assign]

        if isinstance(state['optimiser'], dict):
            self.optimiser = self.init_optimiser(
                self.get_param_groups(None),
                **state.get('optimiser_kwargs', {}),
            )
            self.scheduler = self.init_scheduler(
                self.optimiser,
                **state.get('scheduler_kwargs', {}),
            )
            self.optimiser.load_state_dict(state['optimiser'])
            self.scheduler.load_state_dict(state['scheduler'])
        else:
            warn(
                'Optimiser & scheduler is saved in old non-weights safe format and is '
                'deprecated, please resave the architecture in the new format using '
                'BaseArchitecture.save()',
                DeprecationWarning,
                stacklevel=2,
            )
            self.optimiser = state['optimiser']
            self.scheduler = state['scheduler']

        if 'rng_states' in state:
            self.set_rng_states(state['rng_states'])

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Sets attributes of the architecture, with deprecation warning for save_path.

        Parameters
        ----------
        key : str
            Attribute name
        value : Any
            Attribute value
        """
        if key == 'save_path':
            warn(
                'save_path attribute is deprecated, please use set_save_path method '
                'instead',
                DeprecationWarning,
                stacklevel=2,
            )
            self.set_save_path(str(value), overwrite=True)
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key: str) -> Any:
        """
        Gets attributes of the architecture, with deprecation warning for save_path.

        Parameters
        ----------
        key : str
            Attribute name

        Returns
        -------
        Any
            Attribute value
        """
        if key == 'save_path':
            warn(
                'save_path attribute is deprecated, please use get_save_path method '
                'instead',
                DeprecationWarning,
                stacklevel=2,
            )
            return self._save_path
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

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
            loss = (cast(dict, loss)['total'] if isinstance(loss, dict) else loss) / (i + 1)
            utils.progress_bar(
                i,
                len(loader),
                text=f"Average loss: {loss:.2e}\tTime: {batch_time:.1f}",
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
            [cast(dict, loss)['total']
             if isinstance(loss, dict) else loss for loss in self.losses[0]],
            [cast(dict, loss)['total']
             if isinstance(loss, dict) else loss for loss in self.losses[1]],
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

    def _loss(self, in_data: TensorListLike, target: TensorListLike, extra: Any) -> LossCT:
        """
        Returns the loss as a float & updates network weights if training.

        Parameters
        ----------
        in_data : TensorListLike
            Input data of shape (N,...) and type float, where N is the number of elements
        target : TensorListLike
            Target data of shape (N,...) and type float
        extra : Any
            Extra data from the data loader to pass to the loss function

        Returns
        -------
        LossCT
            Loss or dictionary of losses which can be summed to get the total loss
        """
        key: str
        value: Tensor
        loss: TensorLossCT

        with torch.autocast(
                enabled=self._half,
                dtype=torch.bfloat16 if self._device == torch.device('cpu') else torch.float16,
                device_type=self._device.type):
            try:
                loss = self._loss_func(in_data, target)
                warn(
                    '_loss_func is deprecated, please use _loss_tensor instead',
                    DeprecationWarning,
                    stacklevel=2,
                )
            except DeprecationWarning:
                try:
                    loss = self._loss_tensor(in_data, target, extra)
                except TypeError:
                    warn(
                        '_loss_tensor without extra parameter is deprecated, please update '
                        'the method to include the extra parameter',
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    loss = self._loss_tensor(in_data, target)  # type: ignore[call-arg]

        if isinstance(loss, dict) and 'total' not in loss:
            loss['total'] = self._loss_total(loss)

        self._update(loss['total'] if isinstance(loss, dict) else loss)

        if isinstance(loss, dict):
            return {key: value.item() for key, value in loss.items()}  # type: ignore[return-value]
        return loss.item()  # type: ignore[return-value]

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

    @abstractmethod
    def _loss_tensor(
            self,
            in_data: TensorListLike,
            target: TensorListLike,
            extra: Any) -> TensorLossCT:
        """
        Empty method for child classes to base their loss functions on.

        Parameters
        ----------
        in_data : TensorListLike
            Input data of shape (N,...) and type float, where N is the number of elements
        target : TensorListLike
            Target data of shape (N,...) and type float
        extra : Any
            Extra data from the data loader

        Returns
        -------
        TensorLossCT
            Loss of shape (1) and type float or dictionary of losses of shape (1) and type float
        """

    def _loss_total(self, losses: dict[str, Tensor]) -> Tensor:
        """
        Weighted sum of the loss function terms.

        Parameters
        ----------
        losses : dict[str, Tensor]
            Dictionary of losses of shape (1) and type float

        Returns
        -------
        Tensor
            Total loss of shape (1) and type float
        """
        key: str
        loss_weight: float | Callable[[], float]
        val: Tensor
        total_loss: Tensor = torch.tensor(0., device=self._device)

        for key, val in losses.items():
            loss_weight = self._loss_weights.get(key, 1.0)
            total_loss += val * (loss_weight() if callable(loss_weight) else loss_weight)
        return total_loss

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

    def _step(
            self,
            batch_step: bool,
            epoch: int | float,
            metric: float,
            dataset: DatasetT | None = None) -> None:
        """
        Step method that is called each iteration and each epoch.

        Parameters
        ----------
        batch_step : bool
            If the step is being called during a batch or at the end of an epoch
        epoch : int | float
            Current epoch number, if batch_step is False else current epoch + batch progress
        metric : float
            Loss metric for the iteration or epoch
        dataset : DatasetT, Optional
            Dataset used for training
        """
        loss: float | object
        self._update_scheduler(batch_step=batch_step, epoch=epoch, metrics=metric)

        for loss in self._loss_weights.values():
            if hasattr(loss, 'step'):
                loss.step(epoch)

        if dataset and hasattr(dataset, 'step'):
            dataset.step(epoch)

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
        extra: list[Any]
        batch_loss: LossCT
        target: TensorListLike
        in_data: TensorListLike

        with torch.set_grad_enabled(self._train_state):
            for i, (_, low_dim, high_dim, *extra) in enumerate(loader):
                t_initial = time()
                in_data, target = cast(
                    tuple[TensorListLike, TensorListLike],
                    self._data_loader_translation(
                        data_collation(low_dim, data_field=False).to(self._device),
                        data_collation(high_dim, data_field=False).to(self._device),
                    ),
                )

                try:
                    batch_loss = self._loss(in_data, target, extra[0] if extra else None)
                except TypeError:
                    warn(
                        '_loss without extra parameter is deprecated, please update the '
                        'method to include the extra parameter',
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    batch_loss = self._loss(in_data, target)  # type: ignore[call-arg] # pylint: disable=no-value-for-parameter

                if isinstance(batch_loss, dict) and loss:
                    for key, value in batch_loss.items():
                        if key in loss:
                            loss[key] += value
                        else:
                            loss[key] = value
                elif isinstance(batch_loss, float) and loss:
                    loss += batch_loss
                else:
                    loss = batch_loss

                if self._train_state:
                    self._step(
                        True,
                        self._epoch + (i + 1) / len(loader),
                        batch_loss['total'] if isinstance(batch_loss, dict) else batch_loss,
                        dataset=loader.dataset.dataset
                        if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'dataset')
                        else None,
                    )

                self._batch_print(i, time() - t_initial, loader, batch_loss)

        assert loss

        if isinstance(loss, dict):
            return {key: value / len(loader) for key, value in loss.items()}
        return loss / len(loader)

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

    def _update_epoch(self) -> None:
        """
        Updates architecture epoch.
        """
        self._epoch += 1

    def _update_scheduler(
            self,
            batch_step: bool | None = None,
            *,
            metrics: float | None = None,
            **_: Any) -> None:
        """
        Updates the scheduler for the architecture.

        Parameters
        ----------
        batch_step : bool
            If the step is being called during a batch or at the end of an epoch
        metrics : float | None, Optional
            Loss metric to update ReduceLROnPlateau
        """
        learning_rate: list[float | Tensor]
        new_learning_rate: list[float | Tensor]

        if batch_step is None:
            warn(
                'batch_step is now a required argument for _update_scheduler, please update'
                'the method call',
                DeprecationWarning,
                stacklevel=2,
            )
            batch_step = metrics is not None

        if batch_step and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()
            return
        if batch_step or not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            return

        if metrics is None:
            warn(
                'metrics is a required argument for _update_scheduler when batch_step is '
                'False and scheduler is ReduceLROnPlateau, please update the method call',
                DeprecationWarning,
                stacklevel=2,
            )
            return

        try:
            learning_rate = self.scheduler.get_last_lr()
            self.scheduler.step(metrics)
            new_learning_rate = self.scheduler.get_last_lr()

            if learning_rate[-1] != new_learning_rate[-1]:
                print(f'Learning rate update: {new_learning_rate[-1]:.3e}')
        except AttributeError:
            self.scheduler.step(metrics)

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

    def compile(self, level: Literal['net', 'loss'] = 'net', **kwargs: Any) -> None:
        """
        Compiles the network using torch.compile for faster training and prediction.

        Parameters
        ----------
        level : {'net', 'loss'}
            If 'net', compiles the network, if 'loss', compiles the loss function
        **kwargs
            Optional keyword arguments to pass to torch.compile
        """
        if level == 'net':
            self.net = cast(BaseNetwork, torch.compile(self.net, **kwargs))
        elif level == 'loss':
            self._loss_tensor = torch.compile(self._loss_tensor, **kwargs)  # type: ignore[method-assign] # pylint: disable=attribute-defined-outside-init, method-hidden
        else:
            raise ValueError(f'Invalid compile level ({level}), must be "net" or "loss"')

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
        Gets the device of the architecture.

        Returns
        -------
        torch.device
            Device of the architecture
        """
        return self._device

    def get_epochs(self) -> int:
        """
        Returns the number of epochs the architecture has been trained for.

        Returns
        -------
        int
            Number of epochs
        """
        return self._epoch

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Returns the hyperparameters of the architecture.

        Returns
        -------
        dict[str, Any]
            Hyperparameters of the architecture
        """
        return {
            'mix_precision': self._half,
            'save_freq': self._save_freq,
            'description': self.description,
            'net_name': self.net.name,
            'architecture_name': self.__class__.__name__,
            'version': self.version,
            'verbose': self._verbose,
            'optimiser': self.optimiser.__class__.__name__,
            'scheduler': self.scheduler.__class__.__name__,
            'loss_weights': self._loss_weights,
            'optimiser_kwargs': self._optimiser_kwargs,
            'scheduler_kwargs': self._scheduler_kwargs,
        }

    def get_loader_states(self) -> tuple[Tensor, Tensor] | None:
        """
        Gets the current data loader states.

        Returns
        -------
        tuple[Tensor, Tensor] | None
            Train and validation data loader states
        """
        return self._loader_states

    def get_losses(
            self
    ) -> tuple[list[float], list[float]] | tuple[dict[str, list[float]], dict[str, list[float]]]:
        """
        Returns the training and validation losses as dictionaries of losses if loss is a
        dictionary, else returns losses.

        Returns
        -------
        tuple[list[float], list[float]] | tuple[dict[str, list[float]], dict[str, list[float]]]
            Training and validation losses as dictionaries of lists if loss is a dictionary, else as
            lists
        """
        if not isinstance(self.losses[0][0], dict):
            return self.losses
        return (
            utils.list_dict_convert(self.losses[0], concat=True),
            utils.list_dict_convert(self.losses[1], concat=True),
        )

    @overload
    def get_loss_weights(self, name: str) -> float:
        ...

    @overload
    def get_loss_weights(self) -> dict[str, float]:
        ...

    def get_loss_weights(self, name: str = '') -> float | dict[str, float]:
        """
        Gets the weights for a loss function term or all loss function term weights.

        Parameters
        ----------
        name : str, Optional
            Name of the loss term to get the weight for, if empty string, returns all loss term
            weights

        Returns
        -------
        float | dict[str, float]
            Weight for the specified loss term or all loss term weights
        """
        loss: float | Callable[[], float]
        key: str
        losses: dict[str, float] = {}

        if name and callable(loss := self._loss_weights[name]):
            return loss()
        if name:
            return cast(float, self._loss_weights[name])

        for key, loss in self._loss_weights.items():
            losses[key] = loss() if callable(loss) else loss
        return losses

    def get_param_groups(self, learning_rate: float | tuple[float, ...] | None) -> ParamsT:
        """
        Gets the parameter groups for the optimiser.

        Parameters
        ----------
        learning_rate : float | tuple[float, ...] | None
            Learning rate or learning rates for the parameter groups

        Returns
        -------
        ParamsT
            Parameter groups for the optimiser
        """
        return [{
            'params': self.net.parameters(),
            'lr': learning_rate[0] if isinstance(learning_rate, tuple) else learning_rate or 0,
        }]

    def get_rng_states(self) -> dict[str, Any]:
        """
        Gets the current random number generator states.

        Returns
        -------
        dict[str, Any]
            Random number generator states for numpy, data loaders, pytorch, and cuda
        """
        return {
            'numpy': tuple(state.tolist() if isinstance(state, ndarray) else
                      state for state in np.random.get_state()),
            'loaders': self._loader_states,
            'pytorch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def get_save_path(self) -> str:
        """
        Gets the path to save the architecture.

        Returns
        -------
        str
            Path to save the architecture
        """
        return self._save_path

    def predict(
            self,
            loader: DataLoader[Any],
            *,
            inputs: bool = False,
            path: str = '',
            **kwargs: Any) -> dict[str, NDArrayLike]:
        """
        Generates predictions for the architecture and can save to a file.

        Parameters
        ----------
        loader : DataLoader[Any]
            Data loader to generate predictions for
        inputs : bool, Optional
            If the input data should be returned and saved, default = False
        path : str, Optional
            Path as pkl file to save the predictions if they should be saved
        **kwargs
            Optional keyword arguments to pass to batch_predict

        Returns
        -------
        dict[str, NDArrayLike]
            Prediction IDs, Optional inputs, target values, and predicted values of shape (N,...)
            and type float for dataset of size N
        """
        t_initial: float = time()
        key: str
        ids: tuple[str, ...] | ndarray | Tensor
        low_dim: list[Tensor] | list[Data[Tensor]] | list[DataList[Tensor | Data[Tensor]]]
        high_dim: list[Tensor] | list[Data[Tensor]] | list[DataList[Tensor | Data[Tensor]]]
        data: list[list[NDArrayLike | None]] = []
        data_: dict[str, NDArrayLike] = {}
        transform: (Sequence[BaseTransform[ndarray, ndarray]] |
                    BaseTransform[ndarray, ndarray] | None)
        transforms: dict[
            str,
            Sequence[BaseTransform[ndarray, ndarray]] | BaseTransform[ndarray, ndarray] | None
        ] = {key: transform for key, transform in self.transforms.items()
             if inputs or key != 'inputs'}
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
                data_[key] = datum.apply(cast(
                    ApplyFunc[ndarray | Data[ndarray], ..., ndarray | Data[ndarray]],
                    transform,
                ), back=True).numpy()
                data_[key] = DataList(
                    [trans(val, back=True) for val, trans in zip(
                        datum,
                        transform if isinstance(transform, list) else repeat(transform),
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

    def save(self) -> None:
        """
        Saves the architecture.
        """
        if self._save_path:
            try:
                torch.save(self, self._save_path)
            except (KeyboardInterrupt, SystemExit):
                print('Program interrupted, finishing architecture save...')
                torch.save(self, self._save_path)

    def set_save_freq(self, save_freq: int) -> None:
        """
        Sets the frequency of saving the architecture.

        Parameters
        ----------
        save_freq : int
            Frequency of saving the architecture in epochs
        """
        self._save_freq = save_freq

    @overload
    def set_save_path(self, name: str, states_dir: str, *, overwrite: bool = ...) -> None:
        ...

    @overload
    def set_save_path(self, name: str, *, overwrite: bool = ...) -> None:
        ...

    def set_save_path(self, name: str, states_dir: str = '', *, overwrite: bool = False) -> None:
        """
        Sets the save path for the architecture.

        Parameters
        ----------
        name : str
            Name to save the architecture as or full path to save the architecture
        states_dir : str, Optional
            Directory to save the architecture, if empty name is treated as full path
        overwrite : bool, Optional
            If saving can overwrite an existing save file, if False and file with the same name
            exists, an error will be raised, default = False
        """
        old_save: str = os.path.basename(self._save_path)

        if states_dir:
            self._save_path = utils.save_name(name, states_dir, self.net.name)
        else:
            self._save_path = name if '.pth' in name else f'{name}.pth'

        if old_save == os.path.basename(self._save_path):
            return

        if os.path.exists(self._save_path) and overwrite:
            self._logger.warning(f'{self._save_path} already exists and will be overwritten if '
                                 f'training continues')
        elif os.path.exists(self._save_path):
            raise FileExistsError(f'{self._save_path} already exists and overwrite is False')

    def set_loss_weights(
            self,
            *args: float | Callable[[], float],
            **kwargs: float | Callable[[], float]) -> None:
        """
        Sets the weights for the loss function terms.
        Loss weights passed as positional arguments are set in the order of the architecture loss
        weight keys, while keyword arguments are set by name and will override positional arguments.

        Parameters
        ----------
        *args, **kwargs
            Weights for the loss function terms
        """
        key: str
        value: float | Callable[[], float]

        if not self._loss_weights:
            self._logger.warning('No loss weights to set for this architecture, skipping...')
            return

        for key, value in zip(self._loss_weights, args):
            self._loss_weights[key] = value

        for key, value in kwargs.items():
            if key in self._loss_weights:
                self._loss_weights[key] = value
            else:
                self._logger.warning(f'Loss weight ({key}) not found in architecture loss '
                                     f'weights, skipping...')

    def set_rng_states(self, states: dict[str, Any]) -> None:
        """
        Sets the random number generator states.

        Parameters
        ----------
        states : dict[str, Any]
            Random number generator states for numpy, data loaders, pytorch, and cuda
        """
        cuda_state: Tensor | None
        np.random.set_state(states.pop('numpy', np.random.get_state()))
        self._loader_states = states.pop('loaders', None)
        torch.set_rng_state(states.pop('pytorch', torch.get_rng_state()).cpu())

        if torch.cuda.is_available() and (cuda_state := states.pop('cuda_rng', None)):
            torch.cuda.set_rng_state_all(cuda_state)

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
            The architecture with parameters and buffers moved/cast
        """
        self.net = self.net.to(*args, **kwargs)
        self._param_device(self.optimiser.state, *args, **kwargs)
        self._device, *_ = torch._C._nn._parse_to(*args, **kwargs)  # pylint: disable=protected-access

        if isinstance(self.net, OptimizedModule):
            self.net._orig_mod.to(*args, **kwargs)  # pylint: disable=protected-access
        return self

    def train(self, train: bool) -> None:
        """
        Changes the train/eval state of the architecture.

        Parameters
        ----------
        train : bool
            If the architecture should be in the train state
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

            try:
                self._step(
                    False,
                    i,
                    cast(dict, self.losses[1][-1])['total'] if isinstance(self.losses[1][-1], dict)
                    else self.losses[1][-1],
                    dataset=loaders[0].dataset.dataset
                    if hasattr(loaders[0], 'dataset') and hasattr(loaders[0].dataset, 'dataset')
                    else None,
                )
            except TypeError:
                warn(
                    '_step without dataset parameter is deprecated, please update'
                    'the method to include the dataset parameter',
                    DeprecationWarning,
                    stacklevel=2,
                )
                self._step(
                    False,
                    i,
                    cast(dict, self.losses[1][-1])['total'] if isinstance(self.losses[1][-1], dict)
                    else self.losses[1][-1],
                )

            if self.run:
                self.run.log(
                    {'Train Loss': self.losses[0][-1], 'Validation Loss': self.losses[1][-1]}
                    if isinstance(self.losses[0][-1], float) else
                    {f'Train {key}': val for key, val in self.losses[0][-1].items()} |
                    {f'Validation {key}': val for key, val in self.losses[1][-1].items()}
                )

            # Save training progress
            if loaders[0].generator:
                self._loader_states = (
                    loaders[0].generator.get_state(),
                    loaders[1].generator.get_state(),
                )

            self._update_epoch()

            if self._epoch % self._save_freq == 0 or self._epoch == epochs:
                self.save()

            self._epoch_print(i, epochs, time() - t_initial)

        self.train(False)
        self._plot_active = False
        loss = self._train_val(loaders[1])
        print(f"\nFinal validation loss: "
              f"{cast(dict, loss)['total'] if isinstance(loss, dict) else loss:.3e}")


def load_arch(num: int | str, states_dir: str, arch_name: str, **kwargs: Any) -> BaseArchitecture:
    """
    Loads an architecture from file.

    Parameters
    ----------
    num : int | str
        File number or name of the saved state
    states_dir : str
        Directory to the save files
    arch_name : str
        Name of the architecture
    **kwargs
        Optional keyword arguments to pass to torch.load

    Returns
    -------
    BaseArchitecture
        Saved architecture object
    """
    return torch.load(
        utils.save_name(num, states_dir, arch_name),
        **{'map_location': 'cpu'} | kwargs,
    )


__all__ = ['BaseArchitecture', 'load_arch']
