"""
Base dataset classes for use with BaseNetwork
"""
from typing import Any, overload

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from numpy import ndarray


UNSET = object()


class BaseDatasetMeta(type):
    """
    Automatically creates an index for each sample in the dataset after the dataset has been
    initialised.
    """
    def __call__(cls, *args: Any, **kwargs: Any) -> Dataset:
        """
        Parameters
        ----------
        *args
            Optional arguments to pass to BaseDataset class
        **kwargs
            Optional keyword arguments to pass to BaseDataset class

        Returns
        -------
        Dataset
            Dataset instance from the class that inherited BaseDatasetMeta
        """
        instance: Dataset = super().__call__(*args, **kwargs)

        if hasattr(instance, 'idxs') and instance.idxs.dtype != np.int_:
            raise ValueError(f'idxs attribute already exists and does not have type int '
                             f'({instance.idxs.dtype}), idxs attribute must be reserved for sample '
                             f'index')

        if not hasattr(instance, 'high_dim') or instance.high_dim is UNSET:
            raise ValueError(f'{instance.__class__.__name__} has no high_dim attribute which is '
                             f'required by BaseDatasetMeta for creating idxs attribute')

        if len(instance.idxs) != len(instance.high_dim):
            instance.idxs = np.arange(len(instance.high_dim))

        for attribute in ('extra', 'low_dim', 'high_dim'):
            if (getattr(instance, attribute) is not None and
                    len(getattr(instance, attribute)) != len(instance.idxs)):
                raise ValueError(f'Length of attribute {attribute} ({len(attribute)}) and idxs '
                                 f'({len(instance.idxs)}) does not match')

        return instance


class BaseDataset(Dataset, metaclass=BaseDatasetMeta):
    """
    Base dataset class for use with BaseNetwork

    Attributes
    ----------
    extra : list[Any] | ndarray | None, default = None
        Additional data for each sample in the dataset of length N with shape (N,...) and type Any
    idxs : ndarray
        Index for each sample in the dataset with shape (N) and type int
    low_dim : ndarray | Tensor | None, default = None
        Low dimensional data for each sample in the dataset with shape (N)
    high_dim : ndarray | Tensor | object, default = UNSET
        High dimensional data for each sample in the dataset with shape (N), this is required

    Methods
    -------
    get_low_dim(idx) -> ndarray | Tensor
        Gets a low dimensional sample of the given index
    get_high_dim(idx) -> ndarray | Tensor
        Gets a high dimensional sample of the given index
    get_extra(idx) -> ndarray | Tensor
        Gets extra data for the sample of the given index
    """
    def __init__(self) -> None:
        super().__init__()
        self.extra: list[Any] | ndarray | Tensor | None = None
        self.idxs: ndarray = np.array([0])
        self.low_dim: ndarray | Tensor | None = None
        self.high_dim: ndarray | Tensor | object = UNSET

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(
            self,
            idx: int) -> tuple[int, ndarray | Tensor, ndarray | Tensor, Any]:
        """
        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        tuple[int, ndarray | Tensor, ndarray | Tensor, Any]
            Sample index, low dimensional data, high dimensional data, and extra data
        """
        return self.idxs[idx], self.get_low_dim(idx), self.get_high_dim(idx), self.get_extra(idx)

    def get_low_dim(self, idx: int) -> ndarray | Tensor:
        """
        Gets a low dimensional sample of the given index

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        ndarray | Tensor
            Low dimensional sample
        """
        return torch.tensor(()) if self.low_dim is None else self.low_dim[idx]

    def get_high_dim(self, idx: int) -> ndarray | Tensor:
        """
        Gets a high dimensional sample of the given index

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        ndarray | Tensor
            High dimensional sample
        """
        assert isinstance(self.high_dim, (ndarray, Tensor))
        return self.high_dim[idx]

    def get_extra(self, idx: int) -> Any:
        """
        Gets extra data for the sample of the given index

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        ndarray | Tensor
            Sample extra data
        """
        return torch.tensor(()) if self.extra is None else self.extra[idx]


@overload
def loader_init(
        dataset: BaseDataset,
        batch_size: int = 64,
        ratios: tuple[float] = ...,
        idxs: None = ...,
        **kwargs: Any) -> tuple[DataLoader]: ...

@overload
def loader_init(
        dataset: BaseDataset,
        batch_size: int = 64,
        ratios: None = ...,
        idxs: tuple[ndarray] = ...,
        **kwargs: Any) -> tuple[DataLoader]: ...

@overload
def loader_init(
        dataset: BaseDataset,
        batch_size: int = 64,
        ratios: tuple[float, float] = ...,
        idxs: None = ...,
        **kwargs: Any) -> tuple[DataLoader, DataLoader]: ...

@overload
def loader_init(
        dataset: BaseDataset,
        batch_size: int = 64,
        ratios: tuple[float] = ...,
        idxs: tuple[ndarray] = ...,
        **kwargs: Any) -> tuple[DataLoader, DataLoader]: ...

@overload
def loader_init(
        dataset: BaseDataset,
        batch_size: int = 64,
        ratios: None = ...,
        idxs: tuple[ndarray, ndarray] = ...,
        **kwargs: Any) -> tuple[DataLoader, DataLoader]: ...

@overload
def loader_init(
        dataset: BaseDataset,
        batch_size: int = 64,
        ratios: tuple[float, float, float] = ...,
        idxs: None = ...,
        **kwargs: Any) -> tuple[DataLoader, DataLoader, DataLoader]: ...

@overload
def loader_init(
        dataset: BaseDataset,
        batch_size: int = 64,
        ratios: tuple[float, float] = ...,
        idxs: tuple[ndarray] = ...,
        **kwargs: Any) -> tuple[DataLoader, DataLoader, DataLoader]: ...

@overload
def loader_init(
        dataset: BaseDataset,
        batch_size: int = 64,
        ratios: tuple[float] = ...,
        idxs: tuple[ndarray, ndarray] = ...,
        **kwargs: Any) -> tuple[DataLoader, DataLoader, DataLoader]: ...

@overload
def loader_init(
        dataset: BaseDataset,
        batch_size: int = 64,
        ratios: None = ...,
        idxs: tuple[ndarray, ndarray, ndarray] = ...,
        **kwargs: Any) -> tuple[DataLoader, DataLoader, DataLoader]: ...

def loader_init(
        dataset: Dataset,
        batch_size: int = 64,
        ratios: tuple[float, ...] | None = None,
        idxs: tuple[ndarray, ...] | ndarray | None = None,
        **kwargs: Any) -> tuple[DataLoader, ...]:
    """
    Initialises data loaders from a subset of the dataset with the given ratios.

    Parameters
    ----------
    dataset : Dataset
        Dataset to create data loaders from
    batch_size : int, default = 64
        Batch size when sampling from the data loaders
    ratios : tuple[float, ...] | None, default = (0.8, 0.2)
        Ratios of length M to split up the dataset into subsets, if idxs is provided, dataset will
        first be split up using idxs and ratios will be used on the remaining samples
    idxs : tuple[ndarray, ...] | ndarray | None, default = None
        Dataset indexes for creating the subsets with shape (N,S), where N is the number of subsets
        and S is the number of samples in each subset

    **kwargs : Any
        Optional keyword arguments to pass to DataLoader

    Returns
    -------
    tuple[DataLoader, ...]
        Data loaders for each subset of length N + M
    """
    num: int
    slice_: float | ndarray
    loaders: list[DataLoader] = []
    data_idxs: ndarray = np.arange(len(dataset))
    idxs = list(idxs) if isinstance(idxs, list) or np.ndim(idxs) > 1 else \
        [] if idxs is None else [idxs]
    ratios = ratios or (0.8, 0.2)
    np.random.shuffle(data_idxs)

    if len(ratios) > len(idxs):
        ratios = tuple(np.cumsum(np.array(ratios) / np.sum(ratios)))
    else:
        ratios = ()

    for slice_ in idxs + list(ratios):
        if isinstance(slice_, float):
            num = max(int(len(data_idxs) * slice_), 1)
            slice_ = data_idxs[:num]

        if not np.isin(data_idxs, slice_).any():
            continue

        data_idxs = np.delete(data_idxs, np.isin(data_idxs, slice_))
        loaders.append(DataLoader(
            Subset(dataset, data_idxs[np.isin(data_idxs, slice_)].tolist()),
            batch_size=batch_size,
            **{'shuffle': True} | kwargs,
        ))
    return tuple(loaders)
