"""
Transformations that can be inverted
"""
from types import ModuleType
from typing import Callable, Any, overload

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray

from netloader.utils.utils import ArrayLike


class BaseTransform:
    """
    Base transformation class that other types of transforms build from

    Methods
    -------
    forward(x) -> ArrayLike
        Forward pass of the transformation
    backward(x) -> ArrayLike
        Backwards pass to invert the transformation
    forward_grad(x, uncertainty) -> ArrayLike
        Forward pass of the transformation and uncertainty propagation
    backward_grad(x, uncertainty) -> ArrayLike
        Backwards pass to invert the transformation and uncertainty propagation
    """
    @overload
    def __call__(self, x: ArrayLike, *, back: bool = ...) -> ArrayLike:
        ...

    @overload
    def __call__(
            self,
            x: ArrayLike,
            *,
            back: bool = ...,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        ...

    def __call__(
            self,
            x: ArrayLike,
            back: bool = False,
            uncertainty: ArrayLike | None = None) -> ArrayLike | tuple[ArrayLike, ArrayLike]:
        """
        Calling function returns the forward pass of the transformation

        Parameters
        ----------
        x : (N,...) ArrayLike
            Input array or tensor

        Returns
        -------
        (N,...) ArrayLike
            Transformed array or tensor
        """
        if back and uncertainty is not None:
            return self.backward_grad(x, uncertainty)
        if back:
            return self.backward(x)
        if uncertainty is not None:
            return self.forward_grad(x, uncertainty)
        return self.forward(x)

    def forward(self, x: ArrayLike) -> ArrayLike:
        """
        Forward pass of the transformation

        Parameters
        ----------
        x : (N,...) ArrayLike
            Input array or tensor

        Returns
        -------
        (N,...) ArrayLike
            Transformed array or tensor
        """
        return x

    def backward(self, x: ArrayLike) -> ArrayLike:
        """
        Backwards pass to invert the transformation

        Parameters
        ----------
        x : (N,...) ArrayLike
            Input array or tensor

        Returns
        -------
        (N,...) ArrayLike
            Un-transformed array or tensor
        """
        return x

    def forward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """
        Forward pass of the transformation and uncertainty propagation

        Parameters
        ----------
        x : (N,...) ArrayLike
            Input array or tensor
        uncertainty : (N,...) ArrayLike
            Uncertainty of the input array or tensor

        Returns
        -------
        tuple[(N,...) ArrayLike, (N,...) ArrayLike]
            Transformed array or tensor and transformed uncertainty
        """
        return self(x), uncertainty

    def backward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """
        Backwards pass to invert the transformation and uncertainty propagation

        Parameters
        ----------
        x : (N,...) ArrayLike
            Input array or tensor
        uncertainty : (N,...) ArrayLike
            Uncertainty of the input array or tensor

        Returns
        -------
        (N,...) ArrayLike
            Un-transformed array or tensor and un-transformed uncertainty
        """
        return self(x, back=True), uncertainty


class Log(BaseTransform):
    """
    Logarithm transform

    Attributes
    ----------
    base : float, default = 10
        Base of the logarithm
    idxs : list[int], default = None
        Indices to slice the last dimension to perform the log on

    Methods
    -------
    forward(x) -> ArrayLike
        Forward pass of the transformation
    backward(x) -> ArrayLike
        Backwards pass to invert the transformation
    forward_grad(x, uncertainty) -> ArrayLike
        Forward pass of the transformation and uncertainty propagation
    backward_grad(x, uncertainty) -> ArrayLike
        Backwards pass to invert the transformation and uncertainty propagation
    """
    def __init__(self, base: float = 10, idxs: list[int] | None = None):
        """
        Parameters
        ----------
        base : float, default = 10
            Base of the logarithm
        idxs : list[int], default = None
            Indices to slice the last dimension to perform the log on
        """
        super().__init__()
        self.base: float = base
        self.idxs: list[int] | None = idxs

    def forward(self, x: ArrayLike) -> ArrayLike:
        module: ModuleType = torch if isinstance(x, Tensor) else np
        logs: dict[float, Callable] = {module.e: module.log, 10: module.log10, 2: module.log2}

        if self.base in logs and self.idxs is not None:
            x[..., self.idxs] = logs[self.base](x[..., self.idxs])
        elif self.base in logs:
            x = logs[self.base](x)
        elif self.idxs is not None:
            x[..., self.idxs] = module.log(x[..., self.idxs]) / np.log(self.base)
        else:
            x = module.log(x) / np.log(self.base)
        return x

    def backward(self, x: ArrayLike) -> ArrayLike:
        if self.idxs is not None:
            x[..., self.idxs] = self.base ** x[..., self.idxs]
        else:
            x = self.base ** x
        return x

    def forward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        module: ModuleType = torch if isinstance(x, Tensor) else np

        if self.base == module.e and self.idxs is not None:
            uncertainty[..., self.idxs] *= x ** -1
        elif self.base == module.e:
            uncertainty *= x ** -1
        elif self.idxs is not None:
            uncertainty[..., self.idxs] *= x[..., self.idxs] ** -1 / np.log(self.base)
        else:
            uncertainty *= x ** -1 / np.log(self.base)
        return self(x), uncertainty

    def backward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        module: ModuleType = torch if isinstance(x, Tensor) else np
        x = self(x, back=True)

        if self.idxs is not None:
            uncertainty[..., self.idxs] *= x[..., self.idxs]
        else:
            uncertainty *= x

        if self.base != module.e and self.idxs is not None:
            uncertainty[..., self.idxs] *= np.log(self.base)
        elif self.base != module.e:
            uncertainty *= np.log(self.base)
        return x, uncertainty


class MinClamp(BaseTransform):
    """
    Clamps the minimum value to be the smallest positive value

    Attributes
    ----------
    dim : int, default = None
        Dimension to take the minimum value over

    Methods
    -------
    forward(x) -> ArrayLike
        Forward pass of the transformation
    """
    def __init__(self, dim: int | None = None):
        """
        Parameters
        ----------
        dim : int, default = None
            Dimension to take the minimum value over
        """
        super().__init__()
        self.dim: int | None = dim

    def forward(self, x: ArrayLike) -> ArrayLike:
        kwargs: dict[str, Any]
        module: ModuleType = torch if isinstance(x, Tensor) else np
        min_count: ArrayLike

        if isinstance(x, Tensor):
            kwargs = {'dim': self.dim, 'keepdim': True}
        else:
            kwargs = {'axis': self.dim, 'keepdims': True}

        min_count = module.amin(x[x > 0], **kwargs)
        return module.maximum(x, min_count)


class MultiTransform(BaseTransform):
    """
    Applies multiple transformations

    Attributes
    ----------
    transforms : list[BaseTransform]
        List of transformations

    Methods
    -------
    forward(x) -> ArrayLike
        Forward pass of the transformation
    backward(x) -> ArrayLike
        Backwards pass to invert the transformation
    forward_grad(x, uncertainty) -> ArrayLike
        Forward pass of the transformation and uncertainty propagation
    backward_grad(x, uncertainty) -> ArrayLike
        Backwards pass to invert the transformation and uncertainty propagation
    append(transform)
        Appends a transform to the list of transforms
    """
    def __init__(self, transforms: list[BaseTransform]):
        """
        Parameters
        ----------
        transforms : list[BaseTransform]
            List of transformations
        """
        super().__init__()
        self.transforms = transforms

    def forward(self, x: ArrayLike) -> ArrayLike:
        for transform in self.transforms:
            x = transform(x)
        return x

    def backward(self, x: ArrayLike) -> ArrayLike:
        for transform in self.transforms[::-1]:
            x = transform(x, back=True)
        return x

    def forward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        for transform in self.transforms:
            x, uncertainty = transform(x, uncertainty=uncertainty)
        return x, uncertainty

    def backward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        for transform in self.transforms[::-1]:
            x, uncertainty = transform(x, back=True, uncertainty=uncertainty)
        return x, uncertainty

    def append(self, transform: BaseTransform) -> None:
        """
        Appends a transform to the list of transforms

        Parameters
        ----------
        transform : BaseTransform
            Transform to append to the list of transforms
        """
        self.transforms.append(transform)


class Normalise(BaseTransform):
    """
    Normalises the data to zero mean and unit variance, or between 0 and 1

    Attributes
    ----------
    offset : ndarray
        Offset to subtract from the data
    scale : ndarray
        Scale to divide the data by

    Methods
    -------
    forward(x) -> ArrayLike
        Forward pass of the transformation
    backward(x) -> ArrayLike
        Backwards pass to invert the transformation
    forward_grad(x, uncertainty) -> ArrayLike
        Forward pass of the transformation and uncertainty propagation
    backward_grad(x, uncertainty) -> ArrayLike
        Backwards pass to invert the transformation and uncertainty propagation
    """
    def __init__(
            self,
            data: ArrayLike,
            mean: bool = True,
            dim: int | tuple[int, ...] | None = None):
        """
        Parameters
        ----------
        data : ArrayLike
            Data to normalise
        mean : bool, default = True
            If data should be normalised to zero mean and unit variance, or between 0 and 1
        dim : int | tuple[int, ...] | None, default = None
            Dimensions to normalise over, if None, all dimensions will be normalised over
        """
        super().__init__()
        self.offset: ndarray
        self.scale: ndarray

        if isinstance(data, Tensor):
            data = data.cpu().numpy()

        if mean:
            self.offset = np.mean(data, axis=dim, keepdims=True)
            self.scale = np.std(data, axis=dim, keepdims=True)
        else:
            self.offset = np.amin(data, axis=dim, keepdims=True)
            self.scale = np.amax(data, axis=dim, keepdims=True) - self.offset

    def forward(self, x: ArrayLike) -> ArrayLike:
        if isinstance(x, Tensor):
            return (x - x.new_tensor(self.offset)) / x.new_tensor(self.scale)
        return (x - self.offset) / self.scale

    def backward(self, x: ArrayLike) -> ArrayLike:
        if isinstance(x, Tensor):
            return x * x.new_tensor(self.scale) + x.new_tensor(self.offset)
        return x * self.scale + self.offset

    def forward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        if isinstance(uncertainty, Tensor):
            uncertainty /= uncertainty.new_tensor(self.scale)
        else:
            uncertainty /= self.scale
        return self(x), uncertainty

    def backward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        if isinstance(x, Tensor):
            uncertainty *= uncertainty.new_tensor(self.scale)
        else:
            uncertainty *= self.scale
        return self(x, back=True), uncertainty


class NumpyTensor(BaseTransform):
    """
    Converts Numpy arrays to PyTorch tensors

    Attributes
    ----------
    dtype : dtype, default = float32
        Data type of the tensor

    Methods
    -------
    forward(x) -> ArrayLike
        Forward pass of the transformation
    backward(x) -> ArrayLike
        Backwards pass to invert the transformation
    forward_grad(x, uncertainty) -> ArrayLike
        Forward pass of the transformation and uncertainty propagation
    backward_grad(x, uncertainty) -> ArrayLike
        Backwards pass to invert the transformation and uncertainty propagation
    """
    def __init__(self, dtype: torch.dtype = torch.float32):
        """
        Parameters
        ----------
        dtype : dtype, default = float32
            Data type of the tensor
        """
        super().__init__()
        self.dtype: torch.dtype = dtype

    def forward(self, x: ArrayLike) -> Tensor:
        if isinstance(x, ndarray):
            return torch.from_numpy(x).type(self.dtype)
        return x

    def backward(self, x: ArrayLike) -> ndarray:
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        return x

    def forward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        return self(x), self(uncertainty)

    def backward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        return self(x, back=True), self(uncertainty, back=True)
