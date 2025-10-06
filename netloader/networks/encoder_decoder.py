"""
Classes for encoder, decoder, or autoencoder type architectures
"""
from warnings import warn
from typing import Any, Self, cast

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
    net : nn.Module | Network | CompatibleNetwork
        Neural network
    save_path : str
        Path to the network save file
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
            verbose: str = 'epoch',
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


class Autoencoder(BaseNetwork):
    """
    Network handler for autoencoder type networks.

    Attributes
    ----------
    net : nn.Module | Network | CompatibleNetwork
        Neural network
    reconstruct_loss : float
        Loss weight for the reconstruction MSE loss
    latent_loss : float
        Loss weight for the latent MSE loss
    bound_loss : float
        Loss weight for the latent bounds loss
    kl_loss : float
        Relative weight for KL divergence loss on the latent space
    save_path : str
        Path to the network save file
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
    reconstruct_func : BaseLoss
        Loss function for the reconstruction loss
    latent_func : BaseLoss
        Loss function for the latent loss

    Methods
    -------
    batch_predict(data) -> tuple[NDArrayListLike | None, ...]
        Generates predictions for the given data
    extra_repr() -> str
        Additional representation of the architecture.
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
            verbose: str = 'epoch',
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
        self.reconstruct_loss: float = 1
        self.latent_loss: float = 1e-2
        self.bound_loss: float = 1e-3
        self.kl_loss: float = 1e-1
        self.reconstruct_func: BaseLoss = MSELoss()
        self.latent_func: BaseLoss = MSELoss()

        self.transforms['latent'] = latent_transform
        self.transforms['targets'] = latent_transform

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'reconstruct_loss': self.reconstruct_loss,
            'latent_loss': self.latent_loss,
            'bound_loss': self.bound_loss,
            'kl_loss': self.kl_loss,
            'reconstruct_func': self.reconstruct_func,
            'latent_func': self.latent_func,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.reconstruct_loss = state['reconstruct_loss']
        self.latent_loss = state['latent_loss']
        self.bound_loss = state['bound_loss']
        self.kl_loss = state['kl_loss']
        self.reconstruct_func = state['reconstruct_func']
        self.latent_func = state['latent_func']

    def _loss_tensor(self, in_data: TensorListLike, target: TensorListLike) -> Any:
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
        Tensor
            Loss from the autoencoder's predictions
        """
        loss: Tensor
        latent: Tensor | None = None
        bounds: Tensor = torch.tensor([0., 1.]).to(self._device)
        output: Tensor = self.net(in_data)

        if self.net.checkpoints and isinstance(self.net.checkpoints[-1], DataList):
            raise ValueError(f'Autoencoder networks cannot have multiple latent space tensors '
                             f'({len(self.net.checkpoints[-1])})')
        if self.net.checkpoints:
            latent = cast(Tensor, self.net.checkpoints[-1])

        loss = self.reconstruct_loss * self.reconstruct_func(output, in_data)

        if self.latent_loss and latent is not None:
            loss += self.latent_loss * self.latent_func(latent, target)

        if self.bound_loss and latent is not None:
            loss += self.bound_loss * torch.mean(torch.cat((
                (bounds[0] - latent) ** 2 * (latent < bounds[0]),
                (latent - bounds[1]) ** 2 * (latent > bounds[1]),
            )))

        if self.kl_loss:
            loss += self.kl_loss * self.net.kl_loss
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
        return (f'reconstruct_weight: {self.reconstruct_loss}, '
                f'latent_weight: {self.latent_loss}, '
                f'bound_weight: {self.bound_loss}, '
                f'kl_weight: {self.kl_loss}, '
                f'reconstruct_func: {self.reconstruct_func}, '
                f'latent_func: {self.latent_func}')


class Decoder(BaseNetwork):
    """
    Decoder network for predicting high-dimensional data from low-dimensional inputs.

    Attributes
    ----------
    net : nn.Module | Network | CompatibleNetwork
        Neural network
    save_path : str
        Path to the network save file
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

    Methods
    -------
    extra_repr() -> str
        Additional representation of the architecture.
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


class Encoder(BaseEncoder):
    """
    Encoder network for predicting low-dimensional data from high-dimensional inputs.

    Attributes
    ----------
    net : nn.Module | Network | CompatibleNetwork
        Neural network
    save_path : str
        Path to the network save file
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

    Methods
    -------
    batch_predict(data) -> tuple[ndarray | DataList[ndarray] | None]
        Generates predictions for the given data
    extra_repr() -> str
        Additional representation of the architecture.
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

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)

        if self.classes is not None:
            self.classes = self.classes.to(self._device)

        return self

    def extra_repr(self) -> str:
        """
        Additional representation of the architecture.

        Returns
        -------
        str
            Architecture specific representation
        """
        return f'loss_func: {self._loss_func_}'
