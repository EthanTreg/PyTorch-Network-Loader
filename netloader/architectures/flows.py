"""
Classes that contain normalising flow-based architectures
"""
from typing import Any, Literal, cast

import torch
import numpy as np
from numpy import ndarray
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import ParamsT
from zuko.distributions import NormalizingFlow

from netloader.data import Data
from netloader.utils import label_change
from netloader.loss_funcs import BaseLoss
from netloader.network import BaseNetwork
from netloader.transforms import BaseTransform
from netloader.architectures.base import BaseArchitecture
from netloader.architectures.encoder_decoder import BaseEncoder
from netloader.utils.types import NDArrayLike, TensorListLike, NDArrayListLike, TensorT


class NormFlow(BaseArchitecture):
    """
    Transforms a simple distribution into a distribution that reflects the input data

    Requires last layer to be a normalising flow and will not pass input data through the network.

    Attributes
    ----------
    net : BaseNetwork
        Neural spline flow
    description : str
        Description of the architecture
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
    """
    @staticmethod
    def _data_loader_translation(low_dim: TensorT, high_dim: TensorT) -> tuple[TensorT, TensorT]:
        """
        Orders low and high dimensional tensors from the data loader as inputs and targets for the
        network.

        Parameters
        ----------
        low_dim : TensorT
            Low dimensional tensor from the data loader of shape (N, ...) and type float, where N is
            the batch size
        high_dim : TensorT
            High dimensional tensor from the data loader of shape (N, ...) and type float

        Returns
        -------
        tuple[TensorT, TensorT]
            Input and output target tensors of shape (N, ...) and type float
        """
        return low_dim, high_dim

    def _loss_tensor(self, _: Any, target: TensorListLike, __: Any) -> Tensor:
        """
        Calculates the loss from the flow's predictions.

        Parameters
        ----------
        target : TensorListLike
            Target high dimensional data of shape (N, ...) and type float, where N is the batch size

        Returns
        -------
        Tensor
            Loss from the flow's predictions
        """
        return -self.net().log_prob(target).mean()

    def predict(
            self,
            loader: DataLoader[Any] | None = None,
            *,
            path: str = '',
            num: list[int] | None = None,
            **__: Any) -> dict[str, NDArrayLike]:
        """
        Generates probability distributions for a dataset and can save to a file.

        Parameters
        ----------
        loader : DataLoader[Any] | None, Optional
            Unused, present for compatibility with BaseArchitecture
        path : str, Optional
            Path as pkl file to save the predictions if they should be saved
        num : list[int] | None, Optional
            Number of samples, S, to generate, default = [1e3]

        Returns
        -------
        dict[str, NDArrayLike]
            Predicted distribution, with shape (N,...,S) and type float for dataset of size N
        """
        assert isinstance(self.transforms['preds'], (BaseTransform, type(None)))
        data: dict[str, NDArrayLike]
        samples: ndarray
        transform: BaseTransform | None = self.transforms['preds']

        if num is None:
            num = [int(1e3)]

        samples = self.net().sample(num).moveaxis(0, -1).detach().cpu().numpy()
        data = {'samples': samples if transform is None else transform(samples, back=True)}
        self._save_predictions(path, data)
        return data


class NormFlowEncoder(BaseEncoder):
    """
    Calculates the loss for a network and normalising flow that takes high-dimensional data
    and predicts a low-dimensional data distribution.

    Requires the normalising flow to be the last layer in the network.

    Attributes
    ----------
    description : str
        Description of the architecture
    losses : tuple[list[LossCT], list[LossCT]]
        Architecture training and validation losses as a float or dictionary of losses for each loss
        function
    transforms : dict[str, list[BaseTransform] | BaseTransform | None]
        Keys for the output data from predict and corresponding transforms
    idxs: ndarray | None
        Training data indices with shape (N) and type int, where N is the number of elements in the
        training dataset
    classes : Tensor | None
        Unique classes of shape (C) and type int/float, where C is the number of classes
    optimiser : Optimizer
        Architecture optimiser
    scheduler : LRScheduler
        Optimiser scheduler
    net : BaseNetwork
        Neural network
    """
    _train_flow: bool
    _train_encoder: bool
    _checkpoint: int | None
    _epochs: tuple[int, int]
    net: BaseNetwork[nn.ModuleList]

    def __init__(
            self,
            save_num: int | str,
            states_dir: str,
            net: nn.Module | BaseNetwork[nn.ModuleList],
            *,
            overwrite: bool = False,
            mix_precision: bool = False,
            net_checkpoint: int | None = None,
            description: str = '',
            verbose: Literal['epoch', 'full', 'plot', 'progress', None] = 'epoch',
            train_epochs: tuple[int, int] = (0, -1),
            learning_rate: tuple[float, float] = (1e-3,) * 2,
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
            File number or name to save the flow
        states_dir : str
            Directory to save the architecture and flow
        net : Module | BaseNetwork[ModuleList]
            Normalising flow to predict low-dimensional data distribution
        overwrite : bool, Optional
            If saving can overwrite an existing save file, if True and file with the same name
            exists, an error will be raised, default = False
        mix_precision: bool, Optional
            If mixed precision should be used, default = False
        net_checkpoint : int, Optional
            Network checkpoint for calculating the encoder's loss, if none, will use output from the
            network if output is a Tensor, else no encoder loss will be used
        description : str, Optional
            Description of the architecture training
        verbose : {'full', 'progress', None}
            If details about epoch should be printed ('full'), just a progress bar ('progress'),
            or nothing (None)
        train_epochs : tuple[int, int], Optional
            Epoch when to start training the normalising flow and epoch when to stop training the
            encoder, if both are zero, then flow will be trained from the beginning and encoder will
            not be trained, if both are -1, then flow will never be trained and encoder will always
            be trained, default = (0,-1)
        learning_rate : tuple[float, float], Optional
            Optimiser initial learning rate for encoder and normalising flow, if None, no optimiser
            or scheduler will be set, default = (1e-3,1e-3)
        classes : Tensor, Optional
            Unique classes of shape (C) and type int/float, where C is the number of classes
        loss_func : BaseLoss | None, Optional
            Loss function for the encoder, if None MSELoss will be used if classes is None, else
            CrossEntropyLoss will be used
        in_transform : BaseTransform, Optional
            Transformation for the input data
        optimiser_kwargs : dict[str, Any] | None, Optional
            Optional keyword arguments to pass to init_optimiser
        scheduler_kwargs : dict[str, Any] | None, Optional
            Optional keyword arguments to pass to init_scheduler
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
            loss_func=loss_func,
            transform=transform,
            in_transform=in_transform,
            optimiser_kwargs=optimiser_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        self._checkpoint = net_checkpoint
        self._epochs = train_epochs
        self._train_flow = not self._epochs[0]
        self._train_encoder = bool(self._epochs[-1])
        self._loss_weights = {'flow': 1, 'encoder': 1}
        self.transforms |= {
            'distributions': transform,
            'probs': None,
            'max': transform,
            'meds': transform,
        }
        assert isinstance(self.net, nn.ModuleList)

        if not self._train_encoder:
            self.net.layers[:-1].requires_grad_(False)

        if not self._train_flow:
            self.net.layers[-1].requires_grad_(False)

        self.optimiser = self.init_optimiser(
            self.get_param_groups(learning_rate),
            **optimiser_kwargs or {},
        )
        self.scheduler = self.init_scheduler(self.optimiser, **scheduler_kwargs or {})

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
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._train_flow = state['train_flow']
        self._train_encoder = state['train_encoder']
        self._checkpoint = state['checkpoint']
        self._epochs = state['epochs']
        self._loss_weights = state.get(
            'loss_weights',
            {'flow': state.get('flow_loss', 1), 'encoder': state.get('encoder_loss', 1)},
        )

    def __getattr__(self, item: str) -> Any:
        if item == 'flow_loss':
            return self._loss_weights.get('flow', 0)
        if item == 'encoder_loss':
            return self._loss_weights.get('encoder', 0)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def _loss_tensor(
            self,
            in_data: TensorListLike,
            target: TensorListLike,
            _: Any) -> dict[str, Tensor]:
        """
        Calculates the loss from the network and flow's predictions.

        Parameters
        ----------
        in_data : TensorListLike
            Input high dimensional data of shape (N, ...) and type float, where N is the batch size
        target : TensorListLike
            Target low dimensional data of shape (N, ...) and type float

        Returns
        -------
        dict[str, Tensor]
            Dictionary of losses for encoder and flow
        """
        loss: dict[str, Tensor] = {}
        output: Tensor | NormalizingFlow = self.net(in_data)

        if not isinstance(target, Tensor):
            raise ValueError(f'{self.__class__.__name__} requires target to be a Tensor, got '
                             f'{type(target)}')

        if isinstance(output, NormalizingFlow) and self.get_loss_weights('flow'):
            loss['flow'] = -output.log_prob(target).mean()

        if isinstance(output, NormalizingFlow) and self._checkpoint:
            output = self.net.checkpoints[self._checkpoint]

        # Default shape is (N, L), but cross entropy expects (N)
        if self.get_loss_weights('encoder') and self.classes is not None and \
                isinstance(output, Tensor):
            target = label_change(target.squeeze(), self.classes)

        if self.get_loss_weights('encoder') and isinstance(output, Tensor):
            loss['encoder'] = self._loss_func_(output, target)
        return loss

    def _update_epoch(self) -> None:
        """
        Updates network and flow epoch if they are being trained
        """
        super()._update_epoch()

        if not self._train_flow and self._epoch >= self._epochs[0] != -1:
            self._train_flow = True
            self.net.layers[-1].requires_grad_(True)

        if self._train_encoder and self._epoch >= self._epochs[-1] != -1:
            self._train_encoder = False
            self.net.layers[:-1].requires_grad_(False)

    def batch_predict(
            self,
            data: TensorListLike,
            *,
            num: list[int] | None = None,
            **_: Any) -> tuple[NDArrayListLike | None, ndarray | None]:
        """
        Generates probability distributions for the data batch

        Parameters
        ----------
        data : TensorListLike
            Data of shape (N,...) and type float to generate distributions for, where N is the batch
            size
        num : list[int] | None, Optional
            Number of samples, S, to generate, default = [1e3]

        Returns
        -------
        tuple[NDArrayListLike | None, ndarray | None]
            Architecture output with shape (N,...) and type float and samples of shape (N,S) and
            type float from each probability distribution
        """
        samples: ndarray | None = None
        output: NormalizingFlow | TensorListLike | None = self.net(data)

        if num is None:
            num = [int(1e3)]

        # Generate samples
        if isinstance(output, NormalizingFlow):
            samples = torch.transpose(
                output.sample(num).squeeze(-1),
                0,
                1,
            ).detach().cpu().numpy()
            output = None

        if output is None and self._checkpoint:
            output = self.net.checkpoints[self._checkpoint]

        assert not isinstance(output, NormalizingFlow)
        return (None if output is None else
                cast(NDArrayListLike, output.detach().cpu().numpy()),
                samples)

    def extra_repr(self) -> str:
        """
        Additional representation of the architecture.

        Returns
        -------
        str
            Architecture specific representation
        """
        return f'{super().extra_repr()}, train_epochs: {self._epochs}'

    def get_hyperparams(self) -> dict[str, Any]:
        """
        Get the hyperparameters of the normalising flow encoder.

        Returns
        -------
        dict[str, Any]
            Hyperparameters of the normalising flow encoder
        """
        return super().get_hyperparams() | {
            'checkpoint': self._checkpoint,
            'train_epochs': self._epochs,
        }

    def get_param_groups(self, learning_rate: float | tuple[float, ...] | None) -> ParamsT:
        learning_rate = learning_rate or (0,) * 2

        if isinstance(learning_rate, float):
            learning_rate = (learning_rate,) * 2

        return [
            {'params': self.net.layers[:-1].parameters(), 'lr': learning_rate[0]},
            {'params': self.net.layers[-1:].parameters(), 'lr': learning_rate[1]},
        ]

    def predict(
            self,
            loader: DataLoader[Any],
            *_: Any,
            inputs: bool = False,
            bin_num: int = 100,
            path: str = '',
            num: list[int] | None = None,
            **__: Any) -> dict[str, NDArrayLike]:
        """
        Generates probability distributions for a dataset and can save to a file

        Parameters
        ----------
        loader : DataLoader
            Dataset to generate predictions for
        inputs : bool, Optional
            If the input data should be returned and saved, default = False,
        bin_num : int, Optional
            Number of bins for calculating the probability of the target and maximum of the
            distribution, higher is more precise but requires more samples, default = 100
        path : str, Optional
            Path as a pkl file to save the predictions if they should be saved
        num : list[int] | None, Optional
            Number of samples, S, to generate from the predicted distribution, default = [1e3]

        Returns
        -------
        dict[str, NDArrayLike]
            Prediction IDs of shape (N), Optional inputs, target values, target probability,
            distribution maximum, distribution median of shape (N,...), and predicted distribution
            of shape (N,...,S) and type float for dataset of size N
        """
        probs: list[ndarray] = []
        maxima: list[ndarray] = []
        data: dict[str, NDArrayLike] = super().predict(loader, inputs=inputs, num=num)
        hist: ndarray
        bins: ndarray
        prob: ndarray
        distribution: ndarray
        target: ndarray

        if 'distributions' not in data:
            self._save_predictions(path, data)
            return data

        assert (isinstance(data['distributions'], ndarray) and
                isinstance(data['targets'], (ndarray, Data)))

        for target, distribution in zip(
                data['targets'] if isinstance(data['targets'], ndarray) else data['targets'].data,
                data['distributions']):
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


__all__ = ['NormFlow', 'NormFlowEncoder']
