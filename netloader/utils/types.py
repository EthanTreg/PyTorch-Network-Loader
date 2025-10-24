"""
Type definitions.
"""
from typing import Union, TypeVar, Any, Protocol, TypeAlias, TYPE_CHECKING

from torch import Tensor
from numpy import ndarray

if TYPE_CHECKING:
    from netloader.data import Data, DataList


ArrayLike: TypeAlias = ndarray | Tensor
Param: TypeAlias = dict[str, Union[Tensor, 'Param']]
TensorListLike: TypeAlias = Union[Tensor, 'DataList[Tensor]']
NDArrayListLike: TypeAlias = Union[ndarray, 'DataList[ndarray]']
DataListLike: TypeAlias = Union['DataLike', 'DataList[DataLike]']
DataLike: TypeAlias = Union[ndarray, Tensor, 'Data[ndarray]', 'Data[Tensor]']
TensorLike: TypeAlias = Union[Tensor, 'Data[Tensor]', 'DataList[Tensor | Data[Tensor]]']
NDArrayLike: TypeAlias = Union[ndarray, 'Data[ndarray]', 'DataList[ndarray | Data[ndarray]]']

T = TypeVar('T')
DataT = TypeVar('DataT', bound=DataLike)
ArrayT = TypeVar('ArrayT', bound=ArrayLike)
TensorT = TypeVar('TensorT', bound=TensorLike)
NDArrayT = TypeVar('NDArrayT', bound=NDArrayLike)
DataListT = TypeVar('DataListT', bound=DataListLike)
DatasetT = TypeVar('DatasetT', bound='DatasetProtocol')
TensorListT = TypeVar('TensorListT', bound=TensorListLike)

ArrayCT = TypeVar('ArrayCT', ndarray, Tensor)
ArrayTC = TypeVar('ArrayTC', ndarray, Tensor)
LossCT = TypeVar('LossCT', float, dict[str, float])
TensorLossCT = TypeVar('TensorLossCT', Tensor, dict[str, Tensor])


class DatasetProtocol(Protocol[DataT]):
    """
    Protocol for datasets used in NetLoader

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
    """
    extra: list[Any] | ndarray | Tensor | None
    idxs: ndarray
    low_dim: DataT | None
    high_dim: DataT | None


__all__ = [
    'DatasetProtocol',
    'Param',
    'DataLike',
    'ArrayLike',
    'TensorLike',
    'NDArrayLike',
    'TensorListLike',
    'NDArrayListLike',
    'T',
    'DataT',
    'ArrayT',
    'TensorT',
    'NDArrayT',
    'DatasetT',
    'DataListT',
    'TensorListT',
    'LossCT',
    'ArrayCT',
    'ArrayTC',
    'TensorLossCT',
]
