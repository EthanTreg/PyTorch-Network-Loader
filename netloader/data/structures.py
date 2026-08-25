"""
Data structures for use with NetLoader.
"""
from __future__ import annotations
from types import ModuleType
from itertools import repeat
from typing import (
    Any,
    Self,
    Generic,
    Literal,
    Sequence,
    Iterator,
    Protocol,
    TypeVar,
    cast,
    overload,
)

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray

from netloader.utils.types import DataLike, DataT, ArrayT, ArrayCT, P


InArrayT_contra = TypeVar('InArrayT_contra', bound=DataLike, contravariant=True)
OutArrayT_co = TypeVar('OutArrayT_co', bound=DataLike, covariant=True)


class ApplyFunc(Protocol[InArrayT_contra, P, OutArrayT_co]):
    """
    Protocol for functions that can be applied to Data objects in the apply method of Data and
    DataList.
    """
    def __call__(self, data: InArrayT_contra, /, *args: P.args, **kwargs: P.kwargs) -> OutArrayT_co:
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
    Stores data with uncertainties for uncertainty handling in BaseArchitecture.

    Attributes
    ----------
    data : ArrayCT
        Data
    uncertainty : ArrayCT | None
        Data uncertainty
    """
    def __init__(self, data: ArrayCT, uncertainty: ArrayCT | None = None) -> None:
        """
        Parameters
        ----------
        data : ArrayCT
            Data
        uncertainty : ArrayCT | None, Optional
            Data uncertainty
        """
        self.shape: tuple[int, ...] = tuple(data.shape)
        self.data: ArrayCT = data
        self.uncertainty: ArrayCT | None = uncertainty

    def __eq__(self, other: object) -> bool:
        """
        Checks if two Data objects are equal.

        Parameters
        ----------
        other : object
            Other Data object to compare to

        Returns
        -------
        bool
            If the Data objects are equal
        """
        val: bool | ArrayCT

        if not isinstance(other, Data):
            return NotImplemented
        return bool((self.data == other.data).all() and
                    val if isinstance(val := self.uncertainty == other.uncertainty, bool) else
                    val.all())

    def __getitem__(self, idx: int | slice) -> Data[ArrayCT]:
        """
        Gets a subset of the data and uncertainty.

        Parameters
        ----------
        idx : int | slice
            Index or slice to get

        Returns
        -------
        Data[ArrayCT]
            Data with subset of data and uncertainty
        """
        return Data(
            self.data[idx],
            uncertainty=self.uncertainty[idx] if self.uncertainty is not None else None
        )

    def __len__(self) -> int:
        """
        Returns the number of samples in the data.

        Returns
        -------
        int
            Number of samples in the data
        """
        return len(self.data)

    def __repr__(self) -> str:
        return (f'Data(shape={self.shape}, type={self.data.dtype}, '
                f'uncertainty={self.uncertainty is not None})')

    @staticmethod
    def _apply(
            data: ArrayCT,
            func: str | ApplyFunc[ArrayCT, ..., ArrayCT],
            *args: Any,
            types: tuple[type[DataLike], ...] = (Tensor, ndarray),
            **kwargs: Any) -> ArrayCT:
        """
        Applies a function to the data.

        Parameters
        ----------
        data : ArrayCT
            Data to apply the function to
        func : str | ApplyFunc[ArrayCT, ..., ArrayCT]
            Function, or method if string, to apply to the data and uncertainty
        *args
            Arguments to pass to the function or method
        types : tuple[type[ArrayLike], ...]
            Types to apply the function or method to, if data and uncertainty is not an instance of
            types, then data and uncertainty are returned unchanged, default = (Tensor, ndarray)
        **kwargs
            Keyword arguments to pass to the function or method

        Returns
        -------
        ArrayCT
            Result of applying the function to the data
        """
        if data not in types:
            return data
        if isinstance(func, str):
            return getattr(data, func)(*args, **kwargs)
        return func(data, *args, **kwargs)

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
        data : list[Data[ArrayCT]]
            List of Data objects to collate
        data_field : bool, Optional
            If True, returns a Data object, else returns the collated data as an ArrayLike,
            default = True

        Returns
        -------
        ArrayCT | Data[ArrayCT]
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
            func: str | ApplyFunc[ArrayCT, ..., ArrayCT],
            *args: Any,
            types: tuple[type[DataLike], ...] = (Tensor, ndarray),
            **kwargs: Any) -> Self:
        """
        Applies a function or method to the data and uncertainty.

        Parameters
        ----------
        func : str | ApplyFunc[ArrayCT]
            Function, or method if string, to apply to the data and uncertainty
        *args
            Arguments to pass to the function or method
        types : tuple[type[ArrayLike], ...]
            Types to apply the function or method to, if data and uncertainty is not an instance of
            types, then data and uncertainty are returned unchanged, default = (Tensor, ndarray)
        **kwargs
            Keyword arguments to pass to the function or method

        Returns
        -------
        Self
            Self with function or method applied to the data and uncertainty
        """
        if not isinstance(self.data, types):
            return self

        self.data = self._apply(self.data, func, *args, types=types, **kwargs)
        self.uncertainty = None if self.uncertainty is None else self._apply(
            self.uncertainty,
            func,
            *args,
            types=types,
            **kwargs,
        )
        return self

    def clone(self) -> Data[ArrayCT]:
        """
        Clones the data and uncertainty.

        Identical to Data.copy().

        Returns
        -------
        Data[ArrayCT]
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
        dim : int, Optional
            Dimension to concatenate along, default = 0

        Returns
        -------
        ArrayCT
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
    """
    _names: list[str]
    _data: list[DataT]

    def __init__(
            self,
            data: list[DataT] | dict[str, DataT],
            names: list[str] | None = None) -> None:
        """
        Parameters
        ----------
        data : list[DataT] | dict[str, DataT]
            List or dictionary of Data, Tensor, or ndarray objects
        names : list[str], Optional
            List of names for each data element, only used if data is a list
        """
        if isinstance(data, dict):
            self._names = list(data.keys())
            self._data = list(data.values())
        else:
            self._names = names or []
            self._data = data

        if self._names and len(self._names) != len(self._data):
            raise ValueError(f'Length of names ({len(self._names)}) must match length of data '
                             f'({len(self._data)})')

        if len(self._names) != len(set(self._names)):
            raise ValueError(f'DataList requires unique names, got names: {self._names}')

        if any(len(self._data[0]) != len(datum) for datum in self._data[1:]):
            raise ValueError(f'All elements in DataList must have the same length, got lengths: '
                             f'{[len(datum) for datum in self._data]}')

    def __add__(self, other: DataT | DataList[DataT]) -> DataList[DataT]:
        """
        Adds another DataList or data to the DataList.

        Parameters
        ----------
        other : DataT | DataList[DataT]
            DataList or single element to add to the DataList

        Returns
        -------
        DataList[DataT]
            New DataList with the other DataList or data added to the end of the DataList
        """
        other_names: list[str] = other.get_names() if isinstance(other, DataList) else []

        if len(self) != len(other):
            raise ValueError(f'Lengths of DataList ({len(self)}) and {type(other).__name__} '
                             f'({len(other)}) must be the same')
        if self._names and not isinstance(other, DataList):
            raise ValueError(f'Cannot add {type(other)} to DataList with named elements, use '
                             f'DataList.append')
        if (self._names and not other_names) or (other_names and not self._names):
            raise ValueError(f'Adding DataLists with named elements requires both DataLists to '
                             f'have named elements, got first '
                             f'{"named" if self._names else "unnamed"} and second'
                             f'{"named" if other_names else "unnamed"}')
        return DataList(
            self._data + (list(other) if isinstance(other, DataList) else [other]),
            names=self._names + other_names,
        )

    def __eq__(self, other: object) -> bool:
        """
        Checks if two DataLists are equal.

        Parameters
        ----------
        other : object
            Other DataList to compare to

        Returns
        -------
        bool
            If the DataLists are equal
        """
        val: bool | DataT
        data1: DataT
        data2: DataLike

        if not isinstance(other, DataList):
            return False
        if self.get_names() != other.get_names():
            return False
        return all(val if isinstance(val := data1 == data2, bool) else val.all()
                   for data1, data2 in zip(self.get_data(), other.get_data()))

    @overload
    def __getitem__(self, idx: str) -> DataT: ...

    @overload
    def __getitem__(self, idx: int | slice) -> DataList[DataT]: ...

    def __getitem__(self, idx: int | str | slice) -> DataT | DataList[DataT]:
        """
        Gets a subset of each element in the DataList.

        Parameters
        ----------
        idx: int | str | slice
            Index, name, or slice to get

        Returns
        -------
        DataT | DataList[DataT]
            DataList with subset of each element in the DataList or element if idx is of type str
        """
        return self.get(idx)

    def __iter__(self) -> Iterator[DataT]:
        """
        Iterates over each element in the DataList.

        Returns
        -------
        Iterator[DataT]
            Iterator over each element in the DataList
        """
        return self.iter(list_=True)

    def __len__(self) -> int:
        """
        Returns the number of samples in each element.

        Returns
        -------
        int
            Number of samples in each element
        """
        return self.len(list_=False)

    def __repr__(self) -> str:
        """
        Returns a string representation of the DataList.

        Returns
        -------
        str
            String representation of the DataList
        """
        return (f'DataList(shapes={[tuple(datum.shape) for datum in self._data]}, '
                f'types={[datum.__class__.__name__ for datum in self._data]}'
                f'{f", names={self._names}" if self._names else ""})')

    @staticmethod
    def _apply(
            data: DataT,
            func: str | ApplyFunc[DataT, ..., DataT],
            *args: Any,
            types: tuple[type[DataLike], ...] = (ndarray, Tensor, Data),
            **kwargs: Any) -> DataT:
        """
        Applies a function or method to the data if it is an instance of types, else returns the
        data unchanged.

        Parameters
        ----------
        data : DataT
            Data to apply function to
        func : str | ApplyFunc[ArrayT]
            Function, or method if string, to apply to the data
        *args
            Arguments to pass to the function or method
        types : tuple[type[DataLike], ...]
            Types to apply the function or method to, if data is not an instance of types, then
            data is returned unchanged, default = (ndarray, Tensor, Data)
        **kwargs
            Keyword arguments to pass to the function or method

        Returns
        -------
        DataT
            Result of applying the function or method to the data if it is an instance of types,
            else returns the data unchanged
        """
        if not isinstance(data, types):
            return data
        if isinstance(func, str):
            return getattr(data, func)(*args, **kwargs)
        return func(data, *args, **kwargs)

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
        data_field : bool, Optional
            If True, collates Data elements into Data object, else collates into ArrayLike,
            default = True

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

    def append(self, data: DataT, name: str = '') -> None:
        """
        Appends an element to the DataList.

        Parameters
        ----------
        data : DataT
            Element to append to the DataList
        name : str, Optional
            Name for the element
        """
        if self.len(list_=False) != len(data):
            raise ValueError(f'Element to append must have the same length as the DataList, got '
                             f'lengths: {self.len(list_=False)} and {len(data)}')
        if self._names and not name:
            raise ValueError('DataList contains named elements, name argument is required')
        if name in self._names:
            raise ValueError(f'DataList requires unique names, but got name: {name} and DataList '
                             f'names: {self._names}')

        self._data.append(data)
        self._names.append(name)

    def apply(
            self,
            func: str | Sequence[str | ApplyFunc[DataT, ..., DataT]] | ApplyFunc[DataT, ..., DataT],
            *args: Any,
            types: tuple[type[DataLike], ...] = (Data, Tensor, ndarray),
            **kwargs: Any) -> Self:
        """
        Applies a function or method to each element in the DataList.

        Parameters
        ----------
        func : str | Sequence[str | ApplyFunc[ArrayT]] | ApplyFunc[DataT]
            Function(s), or method(s) if string, to apply to each element in the DataList
        *args
            Arguments to pass to the function or method
        types : tuple[type[DataLike], ...]
            Types to apply the function or method to, if an element is not an instance of types,
            then that element is returned unchanged, default = (Data, Tensor, ndarray)
        **kwargs
            Keyword arguments to pass to the function or method

        Returns
        -------
        Self
            Self with function or method applied to each element in the DataList
        """
        datum: DataT
        func_: str | ApplyFunc[DataT, ..., DataT]

        if isinstance(func, list) and len(func) != len(self._data):
            raise ValueError(f'Length of function list ({len(func)}) must match length of DataList '
                             f'({len(self._data)})')

        self._data = [
            self._apply(datum, func_, *args, types=types, **kwargs)
            for datum, func_ in zip(self._data, func if isinstance(func, list) else repeat(func))
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
        ], names=self._names)

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

    def extend(
            self,
            data: Sequence[DataT] | DataList[DataT],
            names: list[str] | None = None) -> None:
        """
        Extends the DataList by appending elements from the iterable.

        Parameters
        ----------
        data : Sequence[DataT] | DataList[DataT]
            Elements to extend the DataList with
        names : list[str], Optional
            List of names for each data element
        """
        name: str
        datum: DataT

        if self._names and not names:
            raise ValueError('DataList contains named elements, names argument is required')
        if self._names and names and len(names) != len(data):
            raise ValueError(f'Length of names ({len(names)}) must match length of data '
                             f'({len(data)})')

        for name, datum in zip(names or repeat(''), data):
            self.append(datum, name=name)

    @overload
    def get(self, idx: str) -> DataT: ...

    @overload
    def get(self, idx: int, list_: Literal[True]) -> DataT: ...

    @overload
    def get(self, idx: int, list_: Literal[False]) -> DataList[DataT]: ...

    @overload
    def get(self, idx: int, list_: bool = ...) -> DataT | DataList[DataT]: ...

    @overload
    def get(self, idx: slice, list_: bool = ...) -> DataList[DataT]: ...

    def get(
            self,
            idx: int | str | slice,
            list_: bool = False) -> DataT | DataList[DataT]:
        """
        Gets a subset of the DataList or a subset of each element in the DataList.

        Parameters
        ----------
        idx : int | str | slice
            Index, name, or slice to get, if idx is of type str, then ``list_`` argument is ignored
            and the element with the corresponding name is returned
        list_ : bool, Optional
            If True, returns a subset of the DataList, else returns a subset of each element in the
            DataList, default = False

        Returns
        -------
        DataT | DataList[DataT]
            Subset of the DataList or subset of each element in the DataList
        """
        data: DataT

        if isinstance(idx, str):
            if self._names is None:
                raise ValueError("DataList has no names")
            return self._data[self._names.index(idx)]

        if list_ and isinstance(idx, slice):
            return DataList(
                self._data[idx],
                names=names if isinstance(names := self._names[idx], list) else [names],
            )
        if list_ and isinstance(idx, int):
            return self._data[idx]
        return DataList(cast(list[DataT], [data[idx] for data in self]), names=self._names)

    @overload
    def get_data(self, dict_: Literal[False]) -> list[DataT]: ...

    @overload
    def get_data(self, dict_: Literal[True]) -> dict[str, DataT]: ...

    @overload
    def get_data(self, dict_: bool = ...) -> list[DataT] | dict[str, DataT]: ...

    def get_data(self, dict_: bool = False) -> list[DataT] | dict[str, DataT]:
        """
        Gets the underlying data list.

        Parameters
        ----------
        dict_ : bool, Optional
            If True, returns a dictionary of the underlying data list with names as keys, else
            returns a list of the underlying data list, default = False

        Returns
        -------
        list[DataT] | dict[str, DataT]
            Underlying data list
        """
        if dict_:
            if not self._names:
                raise ValueError('Cannot return a dictionary, DataList has no names')
            return dict(zip(self._names, self._data))
        return self._data

    def get_names(self) -> list[str]:
        """
        Gets the names of the elements in the DataList.

        Returns
        -------
        list[str]
            Names of the elements in the DataList
        """
        return self._names

    def insert(self, idx: int, data: DataT, name: str = '') -> None:
        """
        Inserts an element into the DataList at the specified index.

        Parameters
        ----------
        idx : int
            Index to insert the element at
        data : DataT
            Element to insert into the DataList
        name : str, Optional
            Name for the element
        """
        if self.len(list_=False) != len(data):
            raise ValueError(f'Element to insert must have the same length as the DataList, got '
                             f'lengths: {self.len(list_=False)} and {len(data)}')
        if self._names and not name:
            raise ValueError('DataList contains named elements, name argument is required')
        if name in self._names:
            raise ValueError(f'DataList requires unique names, but got name: {name} and DataList '
                             f'names: {self._names}')

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
        list_ : bool, Optional
            If True, iterates over the DataList, else iterates over each element in the DataList,
            default = True

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
        list_ : bool, Optional
            If True, returns the length of the DataList, else returns the length of each element
            in the DataList, default = True

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
        ]), names=self._names)

    def set_names(self, names: list[str]) -> None:
        """
        Sets the names of the elements in the DataList.

        Parameters
        ----------
        names : list[str]
            Names of the elements in the DataList
        """
        if names and len(names) != len(self._data):
            raise ValueError(f'Length of names ({len(names)}) must match length of data '
                             f'({len(self._data)})')

        self._names = names

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
        ]), names=self._names)

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
    data : list[ArrayCT] | list[Data[ArrayCT]] | list[DataList[DataT]] | ArrayCT | DataList[DataT]
        List of ArrayLike, Data, or DataList objects to collate, or if ArrayLike or DataList,
        will return as is, with shape (N,...)
    data_field : bool, Optional
        If Datas should return Data else ArrayLike, default = True

    Returns
    -------
    ArrayCT | Data[ArrayCT] | DataList[ArrayT] | DataList[DataT]
        Collated Data or DataList object, or ArrayLike if data_field is False or input is
        ArrayLike, with shape (N,...)
    """
    if isinstance(data, (Tensor, ndarray, DataList)):
        return data
    if isinstance(data[0], (Tensor, ndarray)):
        return torch.concat(cast(list[Tensor], data)) if isinstance(data[0], Tensor) else \
            np.concat(cast(list[ndarray], data))

    return Data.collate(cast(list[Data], data), data_field=data_field) \
        if isinstance(data[0], Data) else \
        DataList.collate(cast(list[DataList], data), data_field=data_field)


__all__ = ['ApplyFunc', 'Data', 'DataList', 'data_collation']
