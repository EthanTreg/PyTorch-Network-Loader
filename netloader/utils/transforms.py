"""
Transformations that can be inverted
"""
from typing import Callable
from types import ModuleType

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray


class BaseTransform:
    """
    Base transformation class that other types of transforms build from

    Methods
    -------
    forward(x) -> ndarray | Tensor
        Forward pass of the transformation
    backward(x) -> ndarray | Tensor
        Backwards pass to invert the transformation
    """
    def __call__(self, x: ndarray | Tensor) -> ndarray | Tensor:
        """
        Calling function returns the forward pass of the transformation

        Parameters
        ----------
        x : (N,...) ndarray | Tensor
            Input array or tensor

        Returns
        -------
        (N,...) ndarray | Tensor
            Transformed array or tensor
        """
        return self.forward(x)

    def forward(self, x: ndarray | Tensor) -> ndarray | Tensor:
        """
        Forward pass of the transformation

        Parameters
        ----------
        x : (N,...) ndarray | Tensor
            Input array or tensor

        Returns
        -------
        (N,...) ndarray | Tensor
            Transformed array or tensor
        """

    def backward(self, x: ndarray | Tensor) -> ndarray | Tensor:
        """
        Backwards pass to invert the transformation

        Parameters
        ----------
        x : (N,...) ndarray | Tensor
            Input array or tensor

        Returns
        -------
        (N,...) ndarray | Tensor
            Un-transformed array or tensor
        """


class Log(BaseTransform):
    """
    Logarithm transform

    Attributes
    ----------
    base : float, default = e
        Base of the logarithm

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the log
    backward(x) -> Tensor
        Backwards pass to invert the log
    """
    def __init__(self, base: float = torch.e):
        """
        Parameters
        ----------
        base : float, default = e
            Base of the logarithm
        """
        super().__init__()
        self.base: float = base

    def forward(self, x: ndarray | Tensor) -> ndarray | Tensor:
        """
        Forward pass of the log

        Parameters
        ----------
        x : (N,...) ndarray | Tensor
            Input array or tensor

        Returns
        -------
        (N,...) ndarray | Tensor
            Transformed array or tensor
        """
        module: ModuleType = torch if isinstance(x, Tensor) else np
        logs: dict[float, Callable] = {module.e: module.log, 10: module.log10, 2: module.log2}

        if self.base in logs:
            return logs[self.base](x)

        return module.log(x) / np.log(self.base)

    def backward(self, x: ndarray | Tensor) -> ndarray | Tensor:
        """
        Backwards pass to invert the log

        Parameters
        ----------
        x : (N,...) ndarray | Tensor
            Input array or tensor

        Returns
        -------
        (N,...) ndarray | Tensor
            Un-transformed array or tensor
        """
        return self.base ** x


class MultiTransform(BaseTransform):
    """
    Applies multiple transformations

    Attributes
    ----------
    transforms : list[BaseTransform]
        List of transformations

    Methods
    -------
    forward(x) -> ndarray | Tensor
        Forward pass of the transformations
    backward(x) -> ndarray | Tensor
        Backwards pass of the reverse of the transformations
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

    def forward(self, x: ndarray | Tensor) -> ndarray | Tensor:
        """
        Forward pass of the transformations

        Parameters
        ----------
        x : ndarray | Tensor
            Input array or tensor

        Returns
        -------
        ndarray | Tensor
            Transformed array or tensor
        """
        for transform in self.transforms:
            x = transform(x)
        return x

    def backward(self, x: ndarray | Tensor) -> ndarray | Tensor:
        """
        Backwards pass of the reverse of the transformations

        Parameters
        ----------
        x : ndarray | Tensor
            Input array or tensor

        Returns
        -------
        ndarray | Tensor
            Un-transformed array or tensor
        """
        for transform in self.transforms[::-1]:
            x = transform.backward(x)
        return x


class Normalise(BaseTransform):
    """
    Normalises the data to zero mean and unit variance, or between 0 and 1

    Attributes
    ----------
    offset : Tensor
        Offset to subtract from the data
    scale : Tensor
        Scale to divide the data by

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of normalisation
    backward(x) -> Tensor
        Backwards pass to invert the normalisation
    """
    def __init__(
            self,
            data: ndarray | Tensor,
            mean: bool = True,
            dim: int | tuple[int, ...] | None = None):
        """
        Parameters
        ----------
        data : ndarray | Tensor
            Data to normalise
        mean : bool, default = True
            If data should be normalised to zero mean and unit variance, or between 0 and 1
        dim : int | tuple[int, ...] | None, default = None
            Dimensions to normalise over, if None, all dimensions will be normalised over
        """
        super().__init__()
        self.offset: ndarray | Tensor
        self.scale: ndarray | Tensor
        module: ModuleType

        if isinstance(data, Tensor):
            module = torch
            kwargs = {'dim': dim, 'keepdim': True}
        else:
            module = np
            kwargs = {'axis': dim, 'keepdims': True}

        if mean:
            self.offset = module.mean(data, **kwargs)
            self.scale = module.std(data, **kwargs)
        else:
            self.offset = module.amin(data, **kwargs)
            self.scale = module.amax(data, **kwargs) - self.offset

    def forward(self, x: ndarray | Tensor) -> ndarray | Tensor:
        """
        Forward pass of normalisation

        Parameters
        ----------
        x : (N,...) ndarray | Tensor
            Input array or tensor

        Returns
        -------
        (N,...) ndarray | Tensor
            Transformed array or tensor
        """
        if isinstance(x, type(self.offset)):
            return (x - self.offset) / self.scale
        if isinstance(self.offset, ndarray) and isinstance(self.scale, ndarray):
            return (x - torch.from_numpy(self.offset)) / torch.from_numpy(self.scale)
        if isinstance(self.offset, Tensor) and isinstance(self.scale, Tensor):
            return (x - self.offset.numpy()) / self.scale.numpy()
        return (x - self.offset) / self.scale

    def backward(self, x: ndarray | Tensor) -> ndarray | Tensor:
        """
        Backwards pass to invert the normalisation

        Parameters
        ----------
        x : (N,...) ndarray | Tensor
            Input array or tensor

        Returns
        -------
        (N,...) ndarray | Tensor
            Un-transformed array or tensor
        """
        if isinstance(x, type(self.offset)):
            return x * self.scale + self.offset
        if isinstance(self.offset, ndarray) and isinstance(self.scale, ndarray):
            return x * torch.from_numpy(self.scale) + torch.from_numpy(self.offset)
        if isinstance(self.offset, Tensor) and isinstance(self.scale, Tensor):
            return x * self.scale.numpy() + self.offset.numpy()
        return x * self.scale + self.offset


class NumpyTensor(BaseTransform):
    """
    Converts Numpy arrays to PyTorch tensors

    Attributes
    ----------
    dtype : dtype, default = float32
        Data type of the tensor

    Methods
    -------
    forward(x) -> Tensor
        Forward pass to convert Numpy arrays to PyTorch tensors
    backward(x) -> ndarray
        Backwards pass to convert PyTorch tensors to numpy arrays
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

    def forward(self, x: ndarray | Tensor) -> Tensor:
        """
        Forward pass to convert Numpy arrays to PyTorch tensors

        Parameters
        ----------
        x : (N,...) ndarray | Tensor
            Input array or tensor

        Returns
        -------
        (N,...) Tensor
            Converted tensor
        """
        if isinstance(x, ndarray):
            return torch.from_numpy(x).type(self.dtype)
        return x

    def backward(self, x: ndarray | Tensor) -> ndarray:
        """
        Backwards pass to convert PyTorch tensors to numpy arrays

        Parameters
        ----------
        x : (N,...) ndarray | Tensor
            Input array or tensor

        Returns
        -------
        (N,...) ndarray
            Converted array
        """
        if isinstance(x, Tensor):
            return x.detach().cpu().numpy()
        return x
