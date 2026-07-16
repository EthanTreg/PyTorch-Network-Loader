"""
Type definitions.
"""
from __future__ import annotations

from warnings import warn
from typing import Union, TypeVar, Any, Sequence, Protocol, TypeAlias, ParamSpec, TYPE_CHECKING

from numpy import ndarray
from torch import nn, Tensor

if TYPE_CHECKING:
    from netloader.data import Data, DataList


ShapeLike: TypeAlias = list[int] | list[list[int]]
Param: TypeAlias = dict[str, Union[Tensor, 'Param']]
DataLike: TypeAlias = Union[ndarray, Tensor, 'Data[ArrayCT]']
ArrayLike: TypeAlias = ndarray | Tensor
TensorLike: TypeAlias = Union[Tensor, 'Data[Tensor]', 'DataList[Tensor | Data[Tensor]]']
NDArrayLike: TypeAlias = Union[ndarray, 'Data[ndarray]', 'DataList[ndarray | Data[ndarray]]']
DataListLike: TypeAlias = Union[DataLike, 'DataList[DataLike]']
ShapeListLike: TypeAlias = list[int] | Sequence['ShapeListLike']
TensorListLike: TypeAlias = Union[Tensor, 'DataList[Tensor]']
NDArrayListLike: TypeAlias = Union[ndarray, 'DataList[ndarray]']

T = TypeVar('T')
P = ParamSpec('P')
DataT = TypeVar('DataT', bound=DataLike)
ArrayT = TypeVar('ArrayT', bound=ArrayLike)
ModuleT = TypeVar('ModuleT', bound=nn.Module)
TensorT = TypeVar('TensorT', bound=TensorLike)
NDArrayT = TypeVar('NDArrayT', bound=NDArrayLike)
DataListT = TypeVar('DataListT', bound=DataListLike)
TensorListT = TypeVar('TensorListT', bound=TensorListLike)
ModuleListT = TypeVar('ModuleListT', bound=nn.Module | nn.ModuleList)

LossCT = TypeVar('LossCT', float, dict[str, float])
DataCT = TypeVar('DataCT', ndarray, Tensor, 'Data[ndarray]', 'Data[Tensor]')
ArrayCT = TypeVar('ArrayCT', ndarray, Tensor)
TensorLossCT = TypeVar('TensorLossCT', Tensor, dict[str, Tensor])


class DatasetProtocol(Protocol[DataT]):
    """
    Protocol for datasets used in NetLoader

    Attributes
    ----------
    extra : list[Any] | ndarray | None
        Additional data for each sample in the dataset of length N with shape (N,...) and type Any
    idxs : ndarray
        Index for each sample in the dataset with shape (N) and type int
    low_dim : ndarray | Tensor | None
        Low dimensional data for each sample in the dataset with shape (N)
    high_dim : ndarray | Tensor | object
        High dimensional data for each sample in the dataset with shape (N), this is required
    """
    extra: list[Any] | ndarray | Tensor | None
    idxs: ndarray
    low_dim: DataT | None
    high_dim: DataT | None

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns
        -------
        int
            Number of samples in the dataset
        """

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

    def step(self, epoch: float) -> None:
        """
        Step the dataset for each training iteration.

        Parameters
        ----------
        epoch : float
            Current epoch number
        """


DatasetT = TypeVar('DatasetT', bound=DatasetProtocol)


__all__ = [
    'DatasetProtocol',
    'Param',
    'DataLike',
    'ShapeLike',
    'ArrayLike',
    'TensorLike',
    'NDArrayLike',
    'DataListLike',
    'ShapeListLike',
    'TensorListLike',
    'NDArrayListLike',
    'T',
    'P',
    'DataT',
    'ArrayT',
    'ModuleT',
    'TensorT',
    'NDArrayT',
    'DatasetT',
    'DataListT',
    'TensorListT',
    'ModuleListT',
    'LossCT',
    'DataCT',
    'ArrayCT',
    'TensorLossCT',
]

def __getattr__(name: str) -> Any:
    if name == 'ArrayTC':
        warn(
            'ArrayTC is deprecated, use ArrayCT instead',
            DeprecationWarning,
            stacklevel=2,
        )
        return ArrayCT
    raise AttributeError(f"module {__name__} has no attribute {name}")
