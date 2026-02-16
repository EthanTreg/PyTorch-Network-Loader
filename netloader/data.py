"""
Base dataset classes for use with BaseNetwork
"""
from __future__ import annotations
import logging as log
from types import ModuleType
from typing import (
    Any,
    Generic,
    Self,
    Literal,
    Sequence,
    Iterator,
    Callable,
    Protocol,
    cast,
    overload,
)

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from numpy import ndarray

from netloader.utils.types import DataLike, ArrayLike, DataT, ArrayT, DatasetT, DataListT, ArrayCT


class ApplyFunc(Protocol[DataT]):
    """
    Protocol for functions that can be applied to Data objects in the apply method of Data and
    DataList.
    """
    def __call__(self, data: DataT, /, *args: Any, **kwargs: Any) -> DataT:
        """
        Parameters
        ----------
        data : DataT
            Data to apply the function to
        *args
            Optional arguments to pass to the function
        **kwargs
            Optional keyword arguments to pass to the function

        Returns
        -------
        DataT
            Result of applying the function to the data
        """


class Data(Generic[ArrayCT]):
    """
    Stores data with uncertainties for uncertainty handling in BaseNetwork.

    Attributes
    ----------
    data : ArrayTC
        Data
    uncertainty : ArrayCT | None, default = None
        Data uncertainty

    Methods
    -------
    collate(data, data_field=True) -> ArrayCT | Data[ArrayCT]
        Collates a list of Data objects into a single Data object
    apply(func, *args, types, **kwargs) -> Self
        Applies a function or method to the data and uncertainty
    clone() -> Data[ArrayCT]
        Clones the data and uncertainty
    concat(dim=0) -> ArrayCT
        Concatenates data and uncertainty for passing into a network
    copy() -> Data[ArrayCT]
        Copies the data and uncertainty
    cpu() -> Self
        Moves data and uncertainty to CPU if they are Tensors
    detach() -> Self
        Detaches data and uncertainty from the computation graph if they are Tensors
    numpy() -> Data[ndarray]
        Converts data and uncertainty to numpy arrays if they are Tensors
    tensor() -> Data[Tensor]
        Converts data and uncertainty to tensors if they are numpy arrays
    to(*args, **kwargs) -> Self
        Move and/or cast the parameters and buffers
    """
    def __init__(self, data: ArrayCT, uncertainty: ArrayCT | None = None) -> None:
        """
        Parameters
        ----------
        data : ArrayTC
            Data
        uncertainty : ArrayCT | None, default = None
            Data uncertainty
        """
        self.shape: tuple[int, ...] = tuple(data.shape)
        self.data: ArrayCT = data
        self.uncertainty: ArrayCT | None = uncertainty

    def __len__(self) -> int:
        """
        Returns the number of samples in the data.

        Returns
        -------
        int
            Number of samples in the data
        """
        return len(self.data)

    def __getitem__(self, idx: int | slice) -> Data[ArrayCT]:
        """
        Gets a subset of the data and uncertainty.

        Parameters
        ----------
        idx : int | slice
            Index or slice to get

        Returns
        -------
        Data[ArrayTC]
            Data with subset of data and uncertainty
        """
        return Data(
            self.data[idx],
            uncertainty=self.uncertainty[idx] if self.uncertainty is not None else None
        )

    def __repr__(self) -> str:
        return (f'Data(shape={self.data.shape}, type={self.data.__class__.__name__}, '
                f'uncertainty={self.uncertainty is not None})')

    @staticmethod
    @overload
    def collate(data: list[Data[ArrayCT]], *, data_field: Literal[True]) -> Data[ArrayCT]: ...

    @staticmethod
    @overload
    def collate(data: list[Data[ArrayCT]], *, data_field: Literal[False]) -> ArrayCT: ...

    @staticmethod
    @overload
    def collate(data: list[Data[ArrayCT]], *, data_field: bool) -> ArrayCT | Data[ArrayCT]: ...

    @staticmethod
    def collate(
            data: list[Data[ArrayCT]],
            *,
            data_field: bool = True) -> ArrayCT | Data[ArrayCT]:
        """
        Collates a list of Data objects into a single Data object.

        Parameters
        ----------
        data : list[Data[ArrayTC]]
            List of Data objects to collate
        data_field : bool, default = True
            If True, returns a Data object, else returns the collated data as an ArrayLike

        Returns
        -------
        ArrayTC | Data[ArrayTC]
            Collated ArrayLike or Data object
        """
        module: ModuleType = torch if isinstance(data[0].data, Tensor) else np
        new_data: ArrayCT
        datum: Data[Any]

        if data[0].uncertainty is None:
            new_data = module.concat([datum.data for datum in data])
            return Data(new_data) if data_field else new_data

        new_data = module.concat([datum.concat() for datum in data])
        return Data(*new_data.swapaxes(0, 1)) if data_field else new_data

    def apply(
            self,
            func: str | ApplyFunc[ArrayCT],
            *args: Any,
            types: tuple[type[ArrayLike], ...] = (Tensor, ndarray),
            **kwargs: Any) -> Self:
        """
        Applies a function or method to the data and uncertainty.

        Parameters
        ----------
        func : str | ApplyFunc[ArrayCT]
            Function, or method if string, to apply to the data and uncertainty
        *args
            Arguments to pass to the function or method
        types : tuple[type[ArrayLike], ...], default = (Tensor, ndarray)
            Types to apply the function or method to, if data and uncertainty is not an instance of
            types, then data and uncertainty are returned unchanged
        **kwargs
            Keyword arguments to pass to the function or method

        Returns
        -------
        Self
            Self with function or method applied to the data and uncertainty
        """
        if not isinstance(self.data, types):
            return self

        func_: Callable[..., ArrayCT] = lambda data: getattr(data, func)(*args, **kwargs) \
            if isinstance(func, str) else func(data, *args, **kwargs)
        self.data = func_(self.data)
        self.uncertainty = func_(self.uncertainty)
        return self

    def clone(self) -> Data[ArrayCT]:
        """
        Clones the data and uncertainty.

        Identical to Data.copy().

        Returns
        -------
        Data[ArrayTC]
            Cloned Data
        """
        data: ArrayCT
        uncertainty: ArrayCT | None

        if isinstance(self.data, ndarray):
            data = self.data.copy()
            uncertainty = self.uncertainty.copy() if self.uncertainty is not None else None
        else:
            data = self.data.clone()
            uncertainty = self.uncertainty.clone() if self.uncertainty is not None else None
        return Data(data, uncertainty=uncertainty)

    def concat(self, dim: int = 0) -> ArrayCT:
        """
        Concatenates data and uncertainty for passing into a network.

        Parameters
        ----------
        dim : int, default = 0
            Dimension to concatenate along

        Returns
        -------
        ArrayTC
            Data and uncertainty concatenated along the specified dimension if uncertainty is not
            None, else just data
        """
        module: ModuleType = torch if isinstance(self.data, Tensor) else np
        kwargs: dict[str, int] = {'dim' if isinstance(self.data, Tensor) else 'axis': dim}

        if self.uncertainty is not None:
            return module.concat((self.data, self.uncertainty), **kwargs)
        return self.data

    copy = clone

    def cpu(self) -> Self:
        """
        Moves data and uncertainty to CPU if they are Tensors.

        Returns
        -------
        Self
            Self with data and uncertainty on CPU
        """
        if isinstance(self.data, Tensor):
            self.data = self.data.cpu()

        if isinstance(self.uncertainty, Tensor):
            self.uncertainty = self.uncertainty.cpu()
        return self.apply('cpu')

    def detach(self) -> Self:
        """
        Detaches data and uncertainty from the computation graph if they are Tensors.

        Returns
        -------
        Self
            Self with data and uncertainty detached from the computation graph
        """
        if isinstance(self.data, Tensor):
            self.data = self.data.detach()

        if isinstance(self.uncertainty, Tensor):
            self.uncertainty = self.uncertainty.detach()
        return self

    def numpy(self) -> Data[ndarray]:
        """
        Converts data and uncertainty to numpy arrays if they are Tensors.

        Returns
        -------
        Data[ndarray]
            New Data object with data and uncertainty as numpy arrays
        """
        return Data(
            self.data.cpu().numpy() if isinstance(self.data, Tensor) else self.data,
            uncertainty=self.uncertainty.cpu().numpy() if isinstance(self.uncertainty, Tensor) else
            self.uncertainty
        )

    def tensor(self) -> Data[Tensor]:
        """
        Converts data and uncertainty to tensors if they are numpy arrays.

        Returns
        -------
        Data[Tensor]
            New Data object with data and uncertainty as tensors
        """
        return Data(
            torch.from_numpy(self.data) if isinstance(self.data, ndarray) else self.data,
            uncertainty=torch.from_numpy(self.uncertainty) if isinstance(self.uncertainty, ndarray)
            else self.uncertainty
        )

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """
        Move and/or cast the parameters and buffers.

        Parameters
        ----------
        *args
            Arguments to pass to the to method of data and uncertainty
        **kwargs
            Keyword arguments to pass to the to method of data and uncertainty

        Returns
        -------
        Self
            Self with data and uncertainty moved and/or cast
        """
        if isinstance(self.data, Tensor):
            self.data = self.data.to(*args, **kwargs)

        if isinstance(self.uncertainty, Tensor):
            self.uncertainty = self.uncertainty.to(*args, **kwargs)
        return self


class DataList(Generic[DataT]):
    """
    A list that stores tensors, arrays, or Datas and provides batch operations.

    Methods
    -------
    collate(data, data_field=True) -> DataList[DataLike]
        Collates a list of DataList objects into a single DataList object
    append(data) -> None
        Appends an element to the DataList
    apply(func, *args, types, **kwargs) -> Self
        Applies a function or method to each element in the DataList
    clone() -> DataList[DataT]
        Clones the DataList
    copy() -> DataList[DataT]
        Copies the DataList
    cpu() -> Self
        Moves all tensors to CPU
    detach() -> Self
        Detaches all tensors from the computation graph
    extend(data) -> None
        Extends the DataList by appending elements from the iterable
    get(idx, list_=False) -> DataT | list[DataT] | DataList[DataT]
        Gets a subset of the DataList or a subset of each element in the DataList
    insert(idx, data) -> None
        Inserts an element into the DataList at the specified index
    iter(list_=True) -> Iterator[DataT] | Iterator[DataList[DataT]]
        Iterates over each element in the DataList
    len(list_=True) -> int
        Gets the length of the DataList or the length of each element in the DataList
    numpy() -> DataList[ndarray | Data[ndarray]]
        Converts all tensors to numpy arrays
    tensor() -> DataList[Tensor | Data[Tensor]]
        Converts all numpy arrays to tensors
    to(*args, **kwargs) -> DataList[DataLike]
        Move and/or cast the parameters and buffers
    """
    def __init__(self, data: list[DataT]) -> None:
        """
        Parameters
        ----------
        data : list[DataT]
            List of Data, Tensor, or ndarray objects
        """
        self._data: list[DataT] = data

        if any(len(data[0]) != len(datum) for datum in data[1:]):
            raise ValueError(f'All elements in DataList must have the same length, got lengths: '
                             f'{[len(datum) for datum in data]}')

    def __len__(self) -> int:
        """
        Returns the number of elements in the DataList.

        Returns
        -------
        int
            Number of elements in the DataList
        """
        return self.len(list_=False)

    def __getitem__(self, idx: int | slice) -> DataList[DataT]:
        """
        Gets a subset of each element in the DataList.

        Parameters
        ----------
        idx: int | slice
            Index or slice to get

        Returns
        -------
        DataList[DataT]
            DataList with subset of each element in the DataList
        """
        return self.get(idx, list_=False)

    def __iter__(self) -> Iterator[DataT]:
        """
        Iterates over each element in the DataList.

        Returns
        -------
        Iterator[DataT]
            Iterator over each element in the DataList
        """
        return self.iter(list_=True)

    def __repr__(self) -> str:
        return (f'DataList(shapes={[tuple(datum.shape) for datum in self._data]}, '
                f'types={[datum.__class__.__name__ for datum in self._data]})')

    @staticmethod
    @overload
    def collate(
            data: list[DataList[DataT]],
            *,
            data_field: Literal[True]) -> DataList[DataT]: ...

    @staticmethod
    @overload
    def collate(
            data: list[DataList[DataT]],
            *,
            data_field: Literal[False]) -> DataList[ArrayT]: ...

    @staticmethod
    @overload
    def collate(
            data: list[DataList[DataT]],
            *,
            data_field: bool) -> DataList[ArrayT] | DataList[DataT]: ...

    @staticmethod
    def collate(
            data: list[DataList[DataT]],
            *,
            data_field: bool = True) -> DataList[ArrayT] | DataList[DataT]:
        """
        Collates a list of DataList objects into a single DataList object.

        Parameters
        ----------
        data : list[DataList[DataT]]
            List of DataList objects to collate
        data_field : bool, default = True
            If True, collates Data elements into Data object, else collates into ArrayLike

        Returns
        -------
        DataList[ArrayT] | DataList[DataT]
            Collated DataList object
        """
        i: int
        element_data: list[DataLike]
        new_data: list[DataLike] = []
        datum: DataList[DataT]

        for i in range(data[0].len(True)):
            element_data = [datum.get(i, list_=True) for datum in data]
            new_data.append(
                Data.collate(cast(list[Data], element_data), data_field=data_field)
                if isinstance(element_data[0], Data) else
                torch.concat(cast(list[Tensor], element_data))
                if isinstance(element_data[0], Tensor) else
                np.concat(cast(list[ndarray], element_data)),
            )
        return cast(DataList[ArrayT] | DataList[DataT], DataList(new_data))

    def append(self, data: DataT) -> None:
        """
        Appends an element to the DataList.

        Parameters
        ----------
        data : DataT
            Element to append to the DataList
        """
        if self.len(list_=False) != len(data):
            raise ValueError(f'Element to append must have the same length as the DataList, got '
                             f'lengths: {self.len(list_=False)} and {len(data)}')
        self._data.append(data)

    def apply(
            self,
            func: str | ApplyFunc[DataT],
            *args: Any,
            types: tuple[type[DataLike], ...] = (Data, Tensor, ndarray),
            **kwargs: Any) -> Self:
        """
        Applies a function or method to each element in the DataList.

        Parameters
        ----------
        func : str | ApplyFunc[DataT]
            Function, or method if string, to apply to each element in the DataList
        *args
            Arguments to pass to the function or method
        types : tuple[type[DataLike], ...], default = (Data, Tensor, ndarray)
            Types to apply the function or method to, if an element is not an instance of types,
            then that element is returned unchanged
        **kwargs
            Keyword arguments to pass to the function or method

        Returns
        -------
        Self
            Self with function or method applied to each element in the DataList
        """
        datum: DataT
        func_: Callable[..., DataT] = lambda data: data if not isinstance(data, types) else \
            getattr(data, func)(*args, **kwargs) if isinstance(func, str) else \
                func(data, *args, **kwargs)
        self._data = [
            datum.apply(func_) if isinstance(datum, Data) else func_(datum) for datum in self._data
        ]
        return self

    def clone(self) -> DataList[DataT]:
        """
        Clones the DataList.

        Identical to DataList.copy().

        Returns
        -------
        DataList[DataT]
            Cloned DataList
        """
        data: DataT
        return DataList([
            cast(DataT, data.clone() if isinstance(data, Tensor) else data.copy())
            for data in self._data
        ])

    copy = clone

    def cpu(self) -> Self:
        """
        Moves all tensors to CPU

        Returns
        -------
        Self
            Self with all tensors moved to CPU
        """
        return self.apply('cpu', types=(Tensor, Data))

    def detach(self) -> Self:
        """
        Detaches all tensors from the computation graph

        Returns
        -------
        Self
            Self with all tensors detached from the computation graph
        """
        return self.apply('detach', types=(Tensor, Data))

    def extend(self, data: Sequence[DataT] | DataList[DataT]) -> None:
        """
        Extends the DataList by appending elements from the iterable.

        Parameters
        ----------
        data : Sequence[DataT] | DataList[DataT]
            Elements to extend the DataList with
        """
        datum: DataT

        for datum in data:
            self.append(datum)

    @overload
    def get(self, idx: int, list_: Literal[True]) -> DataT: ...

    @overload
    def get(self, idx: slice, list_: Literal[True]) -> list[DataT]: ...

    @overload
    def get(self, idx: int | slice, list_: Literal[False]) -> DataList[DataT]: ...

    @overload
    def get(self, idx: int, list_: bool) -> DataT | DataList[DataT]: ...

    @overload
    def get(self, idx: slice, list_: bool) -> list[DataT] | DataList[DataT]: ...

    def get(self, idx: int | slice, list_: bool = False) -> DataT | list[DataT] | DataList[DataT]:
        """
        Gets a subset of the DataList or a subset of each element in the DataList.

        Parameters
        ----------
        idx : int | slice
            Index or slice to get
        list_ : bool, default = False
            If True, returns a subset of the DataList, else returns a subset of each element in the
            DataList

        Returns
        -------
        DataT | list[DataT] | DataList[DataT]
            Subset of the DataList or subset of each element in the DataList
        """
        data: DataT

        if list_:
            return self._data[idx]
        return DataList(cast(list[DataT], [data[idx] for data in self]))

    def insert(self, idx: int, data: DataT) -> None:
        """
        Inserts an element into the DataList at the specified index.

        Parameters
        ----------
        idx : int
            Index to insert the element at
        data : DataT
            Element to insert into the DataList
        """
        if self.len(list_=False) != len(data):
            raise ValueError(f'Element to insert must have the same length as the DataList, got '
                             f'lengths: {self.len(list_=False)} and {len(data)}')
        self._data.insert(idx, data)

    @overload
    def iter(self, list_: Literal[True]) -> Iterator[DataT]: ...

    @overload
    def iter(self, list_: Literal[False]) -> Iterator[DataList[DataT]]: ...

    @overload
    def iter(self, list_: bool) -> Iterator[DataT] | Iterator[DataList[DataT]]: ...

    def iter(self, list_: bool = True) -> Iterator[DataT] | Iterator[DataList[DataT]]:
        """
        Iterates over each element in the DataList.

        Parameters
        ----------
        list_ : bool, default = True
            If True, iterates over the DataList, else iterates over each element in the DataList

        Returns
        -------
        Iterator[DataT] | Iterator[DataList[DataT]]
            Iterator over each element in the DataList
        """
        i: int

        for i in range(self.len(True)):
            yield self.get(i, list_=list_)

    def len(self, list_: bool = True) -> int:
        """
        Gets the length of the DataList or the length of each element in the DataList.

        Parameters
        ----------
        list_ : bool, default = True
            If True, returns the length of the DataList, else returns the length of each element
            in the DataList

        Returns
        -------
        int
            Length of the DataList or length of each element in the DataList
        """
        return len(self._data) if list_ else len(self._data[0])

    def numpy(self) -> DataList[ndarray | Data[ndarray]]:
        """
        Converts all tensors to numpy arrays.

        Returns
        -------
        DataList[ndarray | Data[ndarray]]
            DataList with all tensors converted to numpy arrays
        """
        data: DataT
        return DataList(cast(list[ndarray | Data[ndarray]], [
            data.cpu().numpy() if hasattr(data, 'cpu') else data for data in self
        ]))

    def tensor(self) -> DataList[Tensor | Data[Tensor]]:
        """
        Converts all numpy arrays to tensors.

        Returns
        -------
        DataList[Tensor | Data[Tensor]]
            DataList with all numpy arrays converted to tensors
        """
        data: DataT
        return DataList(cast(list[Tensor | Data[Tensor]], [
            torch.from_numpy(data) if isinstance(data, ndarray) else
            data.tensor() if isinstance(data, Data) else data for data in self
        ]))

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """
        Move and/or cast the parameters and buffers.

        Parameters
        ----------
        *args
            Arguments to pass to the to method of each element in the DataList
        **kwargs
            Keyword arguments to pass to the to method of each element in the DataList

        Returns
        -------
        Self
            Self with all elements moved and/or cast
        """
        return self.apply('to', *args, types=(Tensor, Data), **kwargs)


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


class BaseDataset(Dataset[Any], Generic[DataListT], metaclass=BaseDatasetMeta):
    """
    Base dataset class for use with BaseNetwork.

    Attributes
    ----------
    extra : list[Any] | ArrayLike | None, default = None
        Additional data for each sample in the dataset of length N with shape (N,...) and type Any
    idxs : ndarray
        Index for each sample in the dataset with shape (N) and type int
    low_dim : DataListT | None, default = None
        Low dimensional data for each sample in the dataset with shape (N,...)
    high_dim : DataListT | None, default = None
        High dimensional data for each sample in the dataset with shape (N,...), this is required

    Methods
    -------
    get_low_dim(idx) -> DataListT
        Gets a low dimensional sample of the given index
    get_high_dim(idx) -> DataListT
        Gets a high dimensional sample of the given index
    get_extra(idx) -> Any
        Gets extra data for the sample of the given index
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


def loader_init(
        dataset: Dataset,
        *,
        return_idxs: bool = False,
        batch_size: int = 64,
        ratios: list[float] | tuple[float, ...] | None = None,
        idxs: list[ndarray] | tuple[ndarray, ...] | ndarray | None = None,
        **kwargs: Any) -> (tuple[DataLoader, ...] |
                           tuple[tuple[DataLoader, ...], tuple[list[int], ...]]):
    """
    Initialises data loaders from a subset of the dataset with the given ratios.

    Parameters
    ----------
    dataset : Dataset
        Dataset to create data loaders from
    return_idxs : bool, default = False
        If the indexes for each data loader should be returned
    batch_size : int, default = 64
        Batch size when sampling from the data loaders
    ratios : list[float] | tuple[float, ...] | None, default = (0.8, 0.2)
        Ratios of length M to split up the dataset into subsets, if idxs is provided, dataset will
        first be split up using idxs and ratios will be used on the remaining samples
    idxs : list[ndarray] | tuple[ndarray, ...] | ndarray | None, default = None
        Dataset indexes for creating the subsets with shape (N,S), where N is the number of subsets
        and S is the number of samples in each subset
    **kwargs : Any
        Optional keyword arguments to pass to DataLoader

    Returns
    -------
    tuple[DataLoader, ...] | tuple[tuple[DataLoader, ...], tuple[list[int], ...]]
        Data loaders for each subset of length N + M and optionally indexes used for each subset
    """
    num: int
    slice_: float | ndarray
    loaders: list[DataLoader] = []
    data_idxs: ndarray = np.arange(len(dataset))  # type: ignore
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
            Subset(dataset, data_idxs[np.isin(data_idxs, slice_)].tolist()),
            batch_size=batch_size,
            **{'shuffle': True} | kwargs,
        ))
        data_idxs = np.delete(data_idxs, np.isin(data_idxs, slice_))

    if return_idxs:
        return tuple(loaders), tuple(loader.dataset.indices for loader in loaders)  # type: ignore
    return tuple(loaders)


@overload
def data_collation(
        data: list[ArrayCT] | ArrayCT,
        *,
        data_field: bool) -> ArrayCT: ...

@overload
def data_collation(
        data: DataList[DataT],
        *,
        data_field: bool) -> DataList[DataT]: ...

@overload
def data_collation(
        data: list[DataList[DataT]],
        *,
        data_field: Literal[True]) -> DataList[DataT]: ...

@overload
def data_collation(
        data: list[DataList[DataT]],
        *,
        data_field: Literal[False]) -> DataList[ArrayT]: ...

@overload
def data_collation(
        data: list[Data[ArrayCT]],
        *,
        data_field: Literal[True]) -> Data[ArrayCT]: ...

@overload
def data_collation(
        data: list[Data[ArrayCT]],
        *,
        data_field: Literal[False]) -> ArrayCT: ...

def data_collation(  # type: ignore
        data: list[ArrayCT] | list[Data[ArrayCT]] | list[DataList[DataT]] | ArrayCT |
              DataList[DataT],
        *,
        data_field: bool = True,
) -> ArrayCT | Data[ArrayCT] | DataList[ArrayT] | DataList[DataT]:
    """
    Collates a list of ArrayLike, Data, or DataList objects into a single object.

    Parameters
    ----------
    data : list[ArrayTC] | list[Data[ArrayTC]] | list[DataList[DataT]] | ArrayTC | DataList[DataT]
        List of ArrayLike, Data, or DataList objects to collate, or if ArrayLike or DataList,
        will return as is, with shape (N,...)
    data_field : bool, default = True
        If Datas should return Data else ArrayLike

    Returns
    -------
    ArrayTC | Data[ArrayTC] | DataList[ArrayT] | DataList[DataT]
        Collated Data or DataList object, or ArrayLike if data_field is False or input is
        ArrayLike, with shape (N,...)
    """
    if isinstance(data, (Tensor, ndarray, DataList)):
        return data
    if isinstance(data[0], (Tensor, ndarray)):
        return torch.concat(cast(list[Tensor], data)) if isinstance(data[0], Tensor) else \
            np.concat(cast(list[ndarray], data))

    # data = cast(list[Data] | list[DataList], data)
    return Data.collate(cast(list[Data], data), data_field=data_field) \
        if isinstance(data[0], Data) else \
        DataList.collate(cast(list[DataList], data), data_field=data_field)
