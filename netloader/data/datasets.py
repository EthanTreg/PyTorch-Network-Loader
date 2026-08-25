"""
Base dataset classes for use with BaseArchitecture.
"""
from __future__ import annotations
import logging as log
from typing import Any, Generic, Literal, Sequence, cast, overload

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from numpy import ndarray

from netloader.utils.types import ArrayLike, DatasetT, DataListT


class BaseDatasetMeta(type):
    """
    Automatically creates an index for each sample in the dataset after the dataset has been
    initialised.
    """
    def __call__(cls: type[DatasetT], *args: Any, **kwargs: Any) -> DatasetT:
        """
        Parameters
        ----------
        cls : type[DatasetT]
            Class that inherited BaseDatasetMeta
        *args
            Optional arguments to pass to BaseDataset class
        **kwargs
            Optional keyword arguments to pass to BaseDataset class

        Returns
        -------
        DatasetT
            Dataset instance from the class that inherited BaseDatasetMeta
        """
        instance: DatasetT = type.__call__(cls, *args, **kwargs)

        if hasattr(instance, 'idxs') and instance.idxs.dtype != np.int_:
            raise ValueError(f'idxs attribute already exists and does not have type int '
                             f'({instance.idxs.dtype}), idxs attribute must be reserved for sample '
                             f'index')

        if not hasattr(instance, 'high_dim') or instance.high_dim is None:
            raise ValueError(f'{instance.__class__.__name__} has no high_dim attribute which is '
                             f'required by BaseDatasetMeta for creating idxs attribute')

        if len(instance.idxs) == 0:
            instance.idxs = np.arange(len(instance.high_dim))
        elif len(instance.idxs) != len(instance.high_dim):
            log.getLogger(__name__).warning(f'Length of idxs ({len(instance.idxs)}) and length of '
                                            f'high_dim ({len(instance.high_dim)}) does not match, '
                                            f'idxs will be sent to a range of high_dim length')
            instance.idxs = np.arange(len(instance.high_dim))

        for attribute in ('extra', 'low_dim', 'high_dim'):
            if (getattr(instance, attribute) is not None and
                    len(getattr(instance, attribute)) != len(instance.idxs)):
                raise ValueError(f'Length of attribute {attribute} '
                                 f'({len(getattr(instance, attribute))}) and idxs '
                                 f'({len(instance.idxs)}) does not match')
        return instance


class BaseDataset(
    Dataset[tuple[int, DataListT, DataListT, Any]],
    Generic[DataListT],
    metaclass=BaseDatasetMeta):
    """
    Base dataset class for use with BaseNetwork.

    Attributes
    ----------
    extra : list[Any] | ArrayLike | None
        Additional data for each sample in the dataset of length N with shape (N,...) and type Any
    idxs : ndarray
        Index for each sample in the dataset with shape (N) and type int
    low_dim : DataListT | None
        Low dimensional data for each sample in the dataset with shape (N,...)
    high_dim : DataListT | None
        High dimensional data for each sample in the dataset with shape (N,...), this is required
    """
    def __init__(self) -> None:
        super().__init__()
        self.extra: list[Any] | ArrayLike | None = None
        self.idxs: ndarray = np.array([], dtype=np.int_)
        self.low_dim: DataListT | None = None
        self.high_dim: DataListT | None = None

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns
        -------
        int
            Number of samples in the dataset
        """
        return len(self.idxs)

    def __getitem__(self, idx: int) -> tuple[int, DataListT, DataListT, Any]:
        """
        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        tuple[int, DataListT, DataListT, Any]
            Sample index, low dimensional data, high dimensional data, and extra data
        """
        return self.idxs[idx], self.get_low_dim(idx), self.get_high_dim(idx), self.get_extra(idx)

    def get_extra(self, idx: int) -> Any:
        """
        Gets extra data for the sample of the given index

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        Any
            Sample extra data
        """
        return torch.tensor(()) if self.extra is None else self.extra[idx]

    def get_high_dim(self, idx: int) -> DataListT:
        """
        Gets a high dimensional sample of the given index

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        DataListT
            High dimensional sample
        """
        assert self.high_dim is not None
        return cast(DataListT, self.high_dim[idx])

    def get_low_dim(self, idx: int) -> DataListT:
        """
        Gets a low dimensional sample of the given index

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        DataListT
            Low dimensional sample
        """
        return cast(DataListT, torch.tensor(()) if self.low_dim is None else self.low_dim[idx])

    def step(self, epoch: float) -> None:
        """
        Step the dataset for each training iteration.

        Parameters
        ----------
        epoch : float
            Current epoch number
        """


@overload
def loader_init(
        dataset: DatasetT,
        *,
        return_idxs: Literal[False],
        batch_size: int = ...,
        ratios: list[float] | tuple[float, ...] | None = ...,
        idxs: list[ndarray] | tuple[ndarray, ...] | ndarray | None = ...,
        **kwargs: Any) -> tuple[DataLoader[tuple[int, DataListT, DataListT, Any]], ...]: ...

@overload
def loader_init(
        dataset: DatasetT,
        *,
        return_idxs: Literal[True],
        batch_size: int = ...,
        ratios: list[float] | tuple[float, ...] | None = ...,
        idxs: list[ndarray] | tuple[ndarray, ...] | ndarray | None = ...,
        **kwargs: Any) -> tuple[
    tuple[DataLoader[tuple[int, DataListT, DataListT, Any]], ...],
    tuple[Sequence[int], ...]]: ...

def loader_init(
        dataset: DatasetT,
        *,
        return_idxs: bool = False,
        batch_size: int = 64,
        ratios: list[float] | tuple[float, ...] | None = None,
        idxs: list[ndarray] | tuple[ndarray, ...] | ndarray | None = None,
        **kwargs: Any,
) -> (tuple[DataLoader[tuple[int, DataListT, DataListT, Any]], ...] |
      tuple[
          tuple[DataLoader[tuple[int, DataListT, DataListT, Any]], ...],
          tuple[Sequence[int], ...]]):
    # pylint: disable=line-too-long
    """
    Initialises data loaders from a subset of the dataset with the given ratios.

    Parameters
    ----------
    dataset : DatasetT
        Dataset to create data loaders from
    return_idxs : bool, Optional
        If the indexes for each data loader should be returned, default = False
    batch_size : int, Optional
        Batch size when sampling from the data loaders, default = 64
    ratios : list[float] | tuple[float, ...] | None, Optional
        Ratios of length M to split up the dataset into subsets, if idxs is provided, dataset will
        first be split up using idxs and ratios will be used on the remaining samples,
        default = (0.8,0.2)
    idxs : list[ndarray] | tuple[ndarray, ...] | ndarray | None, Optional
        Dataset indexes for creating the subsets with shape (N,S), where N is the number of subsets
        and S is the number of samples in each subset
    **kwargs : Any
        Optional keyword arguments to pass to DataLoader

    Returns
    -------
    tuple[DataLoader[tuple[int, DataListT, DataListT, Any]], ...] | tuple[tuple[DataLoader[tuple[int, DataListT, DataListT, Any]], ...], tuple[Sequence[int], ...]]
        Data loaders for each subset of length N + M and optionally indexes used for each subset
    """
    # pylint: enable=line-too-long
    num: int
    slice_: float | ndarray
    loaders: list[DataLoader[tuple[int, DataListT, DataListT, Any]]] = []
    data_idxs: ndarray = np.arange(len(dataset))
    loader: DataLoader[tuple[int, DataListT, DataListT, Any]]
    idxs = [] if idxs is None else list(idxs) \
        if isinstance(idxs, (tuple, list)) or np.ndim(idxs) > 1 else [idxs]
    ratios = (0.8, 1) if ratios is None else list(np.cumsum(np.array(ratios) / np.sum(ratios)))
    np.random.shuffle(data_idxs)

    for slice_ in idxs + list(ratios):
        if isinstance(slice_, (int, float)):
            num = max(int(len(data_idxs) * slice_), 1)
            slice_ = data_idxs[:num]

        if not np.isin(data_idxs, slice_).any():
            continue

        loaders.append(DataLoader(
            Subset(
                cast(Dataset[tuple[int, DataListT, DataListT, Any]], dataset),
                data_idxs[np.isin(data_idxs, slice_)].tolist(),
            ),
            batch_size=batch_size,
            **{'shuffle': True} | kwargs,
        ))
        data_idxs = np.delete(data_idxs, np.isin(data_idxs, slice_))

    if return_idxs:
        return tuple(loaders), tuple(cast(Subset, loader.dataset).indices for loader in loaders)
    return tuple(loaders)


__all__ = ['BaseDataset', 'loader_init']
