"""
Classes for encoder, decoder, or autoencoder type architectures
"""
from warnings import warn
from typing import Any, Self, Literal, cast

import torch
import numpy as np
from numpy import ndarray
from torch import Tensor, nn

from netloader.data import DataList
from netloader.network import Network
from netloader.utils import label_change
from netloader.transforms import BaseTransform
from netloader.networks.base import BaseNetwork
from netloader.loss_funcs import BaseLoss, MSELoss, CrossEntropyLoss
from netloader.utils.types import TensorListLike, NDArrayListLike, TensorT


class BaseEncoder(BaseNetwork):
    """
    Base encoder network for predicting low-dimensional data from high-dimensional inputs.

    Attributes
    ----------
    description : str
        Description of the network
    version : str
        Version of the network
    losses : tuple[list[LossCT], list[LossCT]]
        Network training and validation losses as a float or dictionary of losses for each loss
        function
    transforms : dict[str, list[BaseTransform] | BaseTransform | None]
        Keys for the output data from predict and corresponding transforms
    idxs: ndarray | None
        Training data indices with shape (N) and type int, where N is the number of elements in the
        training dataset
    classes : Tensor | None
        Unique classes of shape (C) and type int/float, where C is the number of classes
    optimiser : optim.Optimizer
        Network optimiser
    scheduler : optim.lr_scheduler.LRScheduler
        Optimiser scheduler
    net : Network | CompatibleNetwork
        Neural network

    Methods
    -------
    extra_repr() -> str
        Additional representation of the architecture
    get_hyperparams() -> dict[str, Any]
        Returns the hyperparameters of the network
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
            learning_rate: float = 1e-3,
            description: str = '',
            verbose: Literal['epoch', 'full', 'plot', 'progress', None] = 'epoch',
            classes: Tensor | None = None,
            loss_func: BaseLoss | None = None,
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
        learning_rate : float, default = 1e-3
            Optimiser initial learning rate, if None, no optimiser or scheduler will be set
        description : str, default = ''
            Description of the network training
        verbose : {'full', 'progress', None}
            If details about epoch should be printed ('full'), just a progress bar ('progress'),
            or nothing (None)
        classes : (C) Tensor, default = None
            Unique classes of size C if using class classification
        loss_func : BaseLoss | None, default = MSELoss | CrossEntropyLoss
            Loss function for the encoder, if None MSELoss will be used if classes is None, else
            CrossEntropyLoss will be used
        transform : BaseTransform, default = None
            Transformation of the low-dimensional data
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
            learning_rate=learning_rate,
            description=description,
            verbose=verbose,
            transform=transform,
            in_transform=in_transform,
            optimiser_kwargs=optimiser_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        self._loss_func_: BaseLoss
        self.classes: Tensor | None

        if classes is None:
            self.classes = classes
            self._loss_func_ = loss_func or MSELoss()
        else:
            self.classes = classes.to(self._device)
            self._loss_func_ = loss_func or CrossEntropyLoss()

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'classes': self.classes, 'loss_func': self._loss_func_}

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.classes = state['classes']

        if 'loss_func' in state:
            self._loss_func_ = state['loss_func']
        else:
            warn(
                f'{self.__class__.__name__} is saved in old non-weights safe format and is '
                'deprecated, please resave the network in the new format using net.save()',
                DeprecationWarning,
                stacklevel=2,
            )

            if self.classes is None:
                self._loss_func_ = MSELoss()
            else:
                self._loss_func_ = CrossEntropyLoss()

    def _loss_tensor(
            self,
            in_data: TensorListLike,
            target: TensorListLike) -> dict[str, Tensor] | Tensor:
        """
        Calculates the loss from the network's predictions.

        Parameters
        ----------
        in_data : TensorListLike
            Input high dimensional data of shape (N, ...) and type float, where N is the batch size
        target : TensorListLike
            Target low dimensional data of shape (N, ...) and type float

        Returns
        -------
        dict[str, Tensor] | Tensor
            Loss from the network's predictions
        """
        # Default shape is (N, L), but cross entropy expects (N)
        if self.classes is not None and isinstance(target, Tensor):
            target = label_change(target.squeeze(), self.classes)
        return self._loss_func_(self.net(in_data), target)

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get the hyperparameters of the encoder.

        Returns
        -------
        dict[str, Any]
            Hyperparameters of the encoder
        """
        return super().get_hyperparams() | {
            'classes': int(self.classes.size(0)) if self.classes is not None else None,
            'loss_func': self._loss_func_.__class__.__name__,
        }

    def extra_repr(self) -> str:
        """
        Additional representation of the architecture.

        Returns
        -------
        str
            Architecture specific representation
        """
        return f'loss_func: {self._loss_func_}'

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)

        if self.classes is not None:
            self.classes = self.classes.to(self._device)
        return self


class Autoencoder(BaseNetwork):
    """
    Network handler for autoencoder type networks.

    Attributes
    ----------
    description : str
        Description of the network
    losses : tuple[list[LossCT], list[LossCT]]
        Network training and validation losses as a float or dictionary of losses for each loss
        function
    transforms : dict[str, list[BaseTransform] | BaseTransform | None]
        Keys for the output data from predict and corresponding transforms
    idxs: ndarray | None
        Training data indices with shape (N) and type int, where N is the number of elements in the
        training dataset
    optimiser : optim.Optimizer
        Network optimiser
    scheduler : optim.lr_scheduler.LRScheduler
        Optimiser scheduler
    net : Network | CompatibleNetwork
        Neural network
    reconstruct_func : BaseLoss
        Loss function for the reconstruction loss
    latent_func : BaseLoss
        Loss function for the latent loss

    Methods
    -------
    batch_predict(data) -> tuple[NDArrayListLike | None, ...]
        Generates predictions for the given data
    extra_repr() -> str
        Additional representation of the architecture
    get_hyperparams() -> dict[str, Any]
        Returns the hyperparameters of the network
    """
    def __init__(
            self,
            save_num: int | str,
            states_dir: str,
            net: nn.Module | Network,
            *,
            overwrite: bool = False,
            mix_precision: bool = False,
            learning_rate: float = 1e-3,
            description: str = '',
            verbose: Literal['epoch', 'full', 'plot', 'progress', None] = 'epoch',
            transform: list[BaseTransform] | BaseTransform | None = None,
            latent_transform: list[BaseTransform] | BaseTransform | None = None,
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
            Description of the network training
        verbose : {'full', 'progress', None}
            If details about epoch should be printed ('full'), just a progress bar ('progress'),
            or nothing (None)
        transform : BaseTransform, default = None
            Transformation applied to the input data
        latent_transform : BaseTransform, default = None
            Transformation applied to the latent space
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
            learning_rate=learning_rate,
            description=description,
            verbose=verbose,
            transform=transform,
            in_transform=transform,
            optimiser_kwargs=optimiser_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        self.reconstruct_func: BaseLoss = MSELoss()
        self.latent_func: BaseLoss = MSELoss()

        self._loss_weights = {'reconstruct': 1, 'latent': 1e-2, 'bound': 1e-3, 'kl': 1e-1}
        self.transforms['latent'] = latent_transform
        self.transforms['targets'] = latent_transform

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'reconstruct_func': self.reconstruct_func,
            'latent_func': self.latent_func,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._loss_weights = state.get('loss_weights', {
            'reconstruct': state.get('reconstruct_loss', 1),
            'latent': state.get('latent_loss', 1),
            'bound': state.get('bound_loss', 1),
            'kl': state.get('kl_loss', 1),
        })
        self.reconstruct_func = state['reconstruct_func']
        self.latent_func = state['latent_func']

    def __getattr__(self, item: str) -> Any:
        if item in {'reconstruct_loss', 'latent_loss', 'bound_loss', 'kl_loss'}:
            warn(
                'Accessing loss weights directly is deprecated, please use '
                'Autoencoder.get_loss_weights() instead',
                DeprecationWarning,
                stacklevel=2,
            )
            return self._loss_weights[item.replace('_loss', '')]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def _loss_tensor(self, in_data: TensorListLike, target: TensorListLike) -> dict[str, Tensor]:
        """
        Calculates the loss from the autoencoder's predictions.

        Parameters
        ----------
        in_data : TensorListLike
            Input high dimensional data of shape (N, ...) and type float, where N is the batch size
        target : TensorListLike
            Latent target low dimensional data of shape (N, ...) and type float

        Returns
        -------
        dict[str, Tensor]
            Loss function terms from the autoencoder's predictions
        """
        loss: dict[str, Tensor] = {}
        latent: Tensor | None = None
        bounds: Tensor = torch.tensor([0., 1.]).to(self._device)
        output: Tensor = self.net(in_data)

        if self.net.checkpoints and isinstance(self.net.checkpoints[-1], DataList):
            raise ValueError(f'Autoencoder networks cannot have multiple latent space tensors '
                             f'({len(self.net.checkpoints[-1])})')
        if self.net.checkpoints:
            latent = cast(Tensor, self.net.checkpoints[-1])

        if self.get_loss_weights('reconstruct'):
            loss['reconstruct'] = self.reconstruct_func(output, in_data)

        if self.get_loss_weights('latent') and latent is not None:
            loss['latent'] = self.latent_func(latent, target)

        if self.get_loss_weights('bound') and latent is not None:
            loss['bound'] = torch.mean(torch.cat((
                (bounds[0] - latent) ** 2 * (latent < bounds[0]),
                (latent - bounds[1]) ** 2 * (latent > bounds[1]),
            )))

        if self.get_loss_weights('kl'):
            loss['kl'] = self.net.kl_loss
        return loss

    def batch_predict(self, data: TensorListLike, **_: Any) -> tuple[NDArrayListLike | None, ...]:
        """
        Generates predictions for the given data batch.

        Parameters
        ----------
        data : TensorListLike
            Data to generate predictions for of shape (N, ...) and type float, where N is the batch
            size

        Returns
        -------
        tuple[NDArrayListLike | None, ...]
            Predictions of shape (N, ...) and type float for the given data
        """
        output: NDArrayListLike = self.net(data).detach().cpu().numpy()
        return (
            output,
            cast(NDArrayListLike, self.net.checkpoints[-1].detach().cpu().numpy()),
            cast(NDArrayListLike, data.detach().cpu().numpy()),
        )

    def extra_repr(self) -> str:
        """
        Additional representation of the architecture.

        Returns
        -------
        str
            Architecture specific representation
        """
        return f'reconstruct_func: {self.reconstruct_func}, latent_func: {self.latent_func}'

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get the hyperparameters of the autoencoder.

        Returns
        -------
        dict[str, Any]
            Hyperparameters of the autoencoder
        """
        return super().get_hyperparams() | {
            'reconstruct_func': self.reconstruct_func.__class__.__name__,
            'latent_func': self.latent_func.__class__.__name__,
        }


class Decoder(BaseNetwork):
    """
    Decoder network for predicting high-dimensional data from low-dimensional inputs.

    Attributes
    ----------
    description : str
        Description of the network
    losses : tuple[list[LossCT], list[LossCT]]
        Network training and validation losses as a float or dictionary of losses for each loss
        function
    transforms : dict[str, list[BaseTransform] | BaseTransform | None]
        Keys for the output data from predict and corresponding transforms
    loss_func : BaseLoss
        Loss function for the reconstructions
    idxs: ndarray | None
        Training data indices with shape (N) and type int, where N is the number of elements in the
        training dataset
    optimiser : optim.Optimizer
        Network optimiser
    scheduler : optim.lr_scheduler.LRScheduler
        Optimiser scheduler
    net : Network | CompatibleNetwork
        Neural network

    Methods
    -------
    extra_repr() -> str
        Additional representation of the architecture
    get_hyperparams() -> dict[str, Any]
        Returns the hyperparameters of the network
    """
    def __init__(
            self,
            save_num: int | str,
            states_dir: str,
            net: nn.Module | Network,
            *,
            overwrite: bool = False,
            mix_precision: bool = False,
            learning_rate: float = 1e-3,
            description: str = '',
            verbose: Literal['epoch', 'full', 'plot', 'progress', None] = 'epoch',
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
            learning_rate=learning_rate,
            description=description,
            verbose=verbose,
            transform=transform,
            in_transform=in_transform,
            optimiser_kwargs=optimiser_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        self.loss_func: BaseLoss = MSELoss()

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'loss_func': self.loss_func}

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.loss_func = state['loss_func']

    @staticmethod
    def _data_loader_translation(low_dim: TensorT, high_dim: TensorT) -> tuple[TensorT, TensorT]:
        """
        Orders low and high dimensional tensors from the data loader as inputs and targets for the
        network.

        Parameters
        ----------
        low_dim : TensorT
            Low dimensional tensor from the data loader of shape (N,...) and type float, where N is
            the batch size
        high_dim : TensorT
            High dimensional tensor from the data loader of shape (N,...) and type float

        Returns
        -------
        tuple[TensorT, TensorT]
            Input and output target tensors of shape (N,...) and type float
        """
        return low_dim, high_dim

    def _loss_tensor(
            self,
            in_data: TensorListLike,
            target: TensorListLike) -> dict[str, Tensor] | Tensor:
        """
        Calculates the loss from the network's predictions.

        Parameters
        ----------
        in_data : TensorListLike
            Input low dimensional data of shape (N, ...) and type float, where N is the batch size
        target : TensorListLike
            Target high dimensional data of shape (N, ...) and type float

        Returns
        -------
        dict[str, Tensor] | Tensor
            Loss from the network's predictions
        """
        return self.loss_func(self.net(in_data), target)

    def extra_repr(self) -> str:
        """
        Additional representation of the architecture.

        Returns
        -------
        str
            Architecture specific representation
        """
        return f'loss_func: {self.loss_func}'

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get the hyperparameters of the decoder.

        Returns
        -------
        dict[str, Any]
            Hyperparameters of the decoder
        """
        return super().get_hyperparams() | {'loss_func': self.loss_func.__class__.__name__}


class Encoder(BaseEncoder):
    """
    Encoder network for predicting low-dimensional data from high-dimensional inputs.

    Attributes
    ----------
    description : str
        Description of the network
    losses : tuple[list[LossCT], list[LossCT]]
        Network training and validation losses as a float or dictionary of losses for each loss
        function
    transforms : dict[str, list[BaseTransform] | BaseTransform | None]
        Keys for the output data from predict and corresponding transforms
    idxs: ndarray | None
        Training data indices with shape (N) and type int, where N is the number of elements in the
        training dataset
    classes : Tensor | None
        Unique classes of shape (C) and type int/float, where C is the number of classes
    optimiser : optim.Optimizer
        Network optimiser
    scheduler : optim.lr_scheduler.LRScheduler
        Optimiser scheduler
    net : Network | CompatibleNetwork
        Neural network

    Methods
    -------
    batch_predict(data) -> tuple[ndarray | DataList[ndarray] | None]
        Generates predictions for the given data
    """
    def batch_predict(self, data: TensorListLike, **_: Any) -> tuple[NDArrayListLike | None, ...]:
        """
        Generates predictions for the given data.

        Parameters
        ----------
        data : TensorListLike
            Data to generate predictions for of shape (N, ...) and type float, where N is the batch
            size

        Returns
        -------
        tuple[NDArrayListLike | None, ...]
            Predictions of shape (N,...) and type float for the given data
        """
        output: NDArrayListLike | None = super().batch_predict(data)[0]
        assert isinstance(output, ndarray)

        if isinstance(self._loss_func_, nn.CrossEntropyLoss):
            output = np.argmax(output, axis=-1, keepdims=True)
        return (output,)
