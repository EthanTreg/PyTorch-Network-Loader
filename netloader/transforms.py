"""
Transformations that can be inverted
"""
from warnings import warn
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
    extra_repr() -> str
        Displays layer parameters when printing the transform
    """
    def __init__(self) -> None:
        # Adds all transform classes to list of safe PyTorch classes when loading saved networks
        torch.serialization.add_safe_globals([self.__class__])

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
        Calling function returns the forward, backwards or uncertainty propagation of the
        transformation

        Parameters
        ----------
        x : ArrayLike
            Input array or tensor of shape (N,...), where N is the number of elements
        back : bool, default = False
            If the inverse transformation should be applied
        uncertainty : ArrayLike, default = None
            Corresponding uncertainties for the input data for uncertainty propagation of shape
            (N,...)

        Returns
        -------
        ArrayLike | tuple[ArrayLike, ArrayLike]
            Transformed array or tensor of shape (N,...) and propagated uncertainties of shape
            (N,...)  if provided
        """
        if back and uncertainty is not None:
            return self.backward_grad(x, uncertainty)
        if back:
            return self.backward(x)
        if uncertainty is not None:
            return self.forward_grad(x, uncertainty)
        return self.forward(x)

    def __repr__(self) -> str:
        """
        Representation of the transformation

        Returns
        -------
        str
            Representation string
        """
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the transformation for pickling

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the transformation
        """
        return {}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the transformation for pickling

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the transformation
        """

    def forward(self, x: ArrayLike) -> ArrayLike:
        """
        Forward pass of the transformation

        Parameters
        ----------
        x : ArrayLike
            Input array or tensor of shape (N,...), where N is the number of elements

        Returns
        -------
        ArrayLike
            Transformed array or tensor of shape (N,...)
        """
        return x

    def backward(self, x: ArrayLike) -> ArrayLike:
        """
        Backwards pass to invert the transformation

        Parameters
        ----------
        x : ArrayLike
            Input array or tensor of shape (N,...), where N is the number of elements

        Returns
        -------
        ArrayLike
            Untransformed array or tensor of shape (N,...)
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
        x : ArrayLike
            Input array or tensor of shape (N,...), where N is the number of elements
        uncertainty : ArrayLike
            Uncertainty of the input array or tensor of shape (N,...)

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            Transformed array or tensor of shape (N,...) and transformed uncertainty of shape
            (N,...)
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
        x : ArrayLike
            Input array or tensor of shape (N,...), where N is the number of elements
        uncertainty : ArrayLike
            Uncertainty of the input array or tensor of shape (N,...)

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            Untransformed array or tensor of shape (N,...) and untransformed uncertainty of shape
            (N,...)
        """
        return self(x, back=True), uncertainty

    def extra_repr(self) -> str:
        """
        Additional representation of the transformation

        Returns
        -------
        str
            Transform specific representation
        """
        return ''


class Index(BaseTransform):
    """
    Slices the input along a given dimension assuming the input meets the required shape

    Methods
    -------
    forward(x) -> ArrayLike
        Forward pass of the transformation
    forward_grad(x, uncertainty) -> ArrayLike
        Forward pass of the transformation and uncertainty propagation
    extra_repr() -> str
        Displays layer parameters when printing the transform
    """
    def __init__(
            self,
            dim: int = -1,
            in_shape: tuple[int, ...] | None = None,
            slice_: slice = slice(None)) -> None:
        """
        Parameters
        ----------
        dim : int, default = -1
            Dimension to slice over
        in_shape : tuple[int, ...] | None, default = None
            Target shape ignoring batch size so that the slice only occurs if the input has the
            same shape to prevent repeated slicing, if any dimension has a shape of -1, then the
            size of the dimension will be ignored
        slice_ : slice, default = slice(None)
            Slicing object
        """
        super().__init__()
        self._shape: tuple[int, ...] = tuple(in_shape or ())
        self._slice: list[slice] = [slice(None)] * (len(self._shape) or 1)
        self._slice[dim] = slice_

    def __getstate__(self) -> dict[str, Any]:
        return {'in_shape': self._shape, 'slice': self._slice}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._shape = state['in_shape']
        self._slice = state['slice']

    def forward(self, x: ArrayLike) -> ArrayLike:
        idxs: ndarray = np.array(np.array(self._shape) != -1)

        if np.array(np.array(x.shape[1:])[idxs] != np.array(self._shape)[idxs]).any():
            return super().forward(x)
        return x[:, *self._slice]

    def forward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        return self(x), self(uncertainty)

    def extra_repr(self) -> str:
        return f'in_shape: {self._shape}, slice: {self._slice}'


class Log(BaseTransform):
    """
    Logarithm transform

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
    extra_repr() -> str
        Displays layer parameters when printing the transform
    """
    def __init__(self, base: float = 10, idxs: list[int] | None = None) -> None:
        """
        Parameters
        ----------
        base : float, default = 10
            Base of the logarithm
        idxs : list[int], default = None
            Indices to slice the last dimension to perform the log on
        """
        super().__init__()
        self._base: float = base
        self._idxs: list[int] | None = idxs

    def __getstate__(self) -> dict[str, Any]:
        return {'base': self._base, 'idxs': self._idxs}

    def __setstate__(self, state: dict[str, Any]) -> None:
        try:
            self._base = state['base']
            self._idxs = state['idxs']
        except KeyError:
            self._base = state['_base']
            self._idxs = state['_idxs']
            warn(f'{self.__class__.__name__} transform is saved in an old format and is '
                 f'deprecated, please resave the transform', DeprecationWarning, stacklevel=2)

    def forward(self, x: ArrayLike) -> ArrayLike:
        module: ModuleType = torch if isinstance(x, Tensor) else np
        logs: dict[float, Callable] = {module.e: module.log, 10: module.log10, 2: module.log2}

        if self._base in logs and self._idxs is not None:
            x[..., self._idxs] = logs[self._base](x[..., self._idxs])
        elif self._base in logs:
            x = logs[self._base](x)
        elif self._idxs is not None:
            x[..., self._idxs] = module.log(x[..., self._idxs]) / np.log(self._base)
        else:
            x = module.log(x) / np.log(self._base)
        return x

    def backward(self, x: ArrayLike) -> ArrayLike:
        if self._idxs is not None:
            x[..., self._idxs] = self._base ** x[..., self._idxs]
        else:
            x = self._base ** x
        return x

    def forward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        module: ModuleType = torch if isinstance(x, Tensor) else np

        if self._base == module.e and self._idxs is not None:
            uncertainty[..., self._idxs] /= x[..., self._idxs]
        elif self._base == module.e:
            uncertainty /= x
        elif self._idxs is not None:
            uncertainty[..., self._idxs] /= x[..., self._idxs] * np.log(self._base)
        else:
            uncertainty /= x * np.log(self._base)
        return self(x), module.abs(uncertainty)

    def backward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        module: ModuleType = torch if isinstance(x, Tensor) else np
        x = self(x, back=True)

        if self._idxs is not None:
            uncertainty[..., self._idxs] *= x[..., self._idxs]
        else:
            uncertainty *= x

        if self._base != module.e and self._idxs is not None:
            uncertainty[..., self._idxs] *= np.log(self._base)
        elif self._base != module.e:
            uncertainty *= np.log(self._base)
        return x, module.abs(uncertainty)

    def extra_repr(self) -> str:
        return f'base: {self._base}, idxs: {self._idxs}'


class MinClamp(BaseTransform):
    """
    Clamps the minimum value to be the smallest positive value

    Methods
    -------
    forward(x) -> ArrayLike
        Forward pass of the transformation
    extra_repr() -> str
        Displays layer parameters when printing the transform
    """
    def __init__(self, dim: int | None = None, idxs: list[int] | None = None) -> None:
        """
        Parameters
        ----------
        dim : int, default = None
            Dimension to take the minimum value over
        idxs : list[int], default = None
            Indices to slice the last dimension to perform the min clamp on
        """
        super().__init__()
        self._dim: int | None = dim
        self._idxs: list[int] | None = idxs

    def __getstate__(self) -> dict[str, Any]:
        return {'dim': self._dim, 'idxs': self._idxs}

    def __setstate__(self, state: dict[str, Any]) -> None:
        try:
            self._dim = state['dim']
            self._idxs = state['idxs']
        except KeyError:
            self._dim = state['_dim']
            self._idxs = state['_idxs']
            warn(f'{self.__class__.__name__} transform is saved in an old format and is '
                 f'deprecated, please resave the transform', DeprecationWarning, stacklevel=2)

    def forward(self, x: ArrayLike) -> ArrayLike:
        kwargs: dict[str, Any]
        module: ModuleType = torch if isinstance(x, Tensor) else np
        x_clamp: ArrayLike
        min_count: ArrayLike

        if isinstance(x, Tensor):
            kwargs = {'dim': self._dim, 'keepdim': True}
        else:
            kwargs = {'axis': self._dim, 'keepdims': True}

        if self._idxs is None:
            min_count = module.amin(module.where(x > 0, x, module.max(x)), **kwargs)
            x = module.maximum(x, min_count)
        else:
            x_clamp = x[..., self._idxs]
            min_count = module.amin(module.where(
                x_clamp > 0,
                x_clamp,
                module.max(x_clamp),
            ), **kwargs)
            x[..., self._idxs] = module.maximum(x_clamp, min_count)
        return x

    def extra_repr(self) -> str:
        return f'dim: {self._dim}, idxs: {self._idxs}'


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
    extra_repr() -> str
        Displays layer parameters when printing the transform
    """
    def __init__(self, *args: BaseTransform) -> None:
        """
        Parameters
        ----------
        args : BaseTransform
            Transformations
        """
        super().__init__()
        self.transforms: list[BaseTransform]

        if isinstance(args[0], list):
            warn(
                'List of transforms is deprecated, pass transforms as arguments directly',
                DeprecationWarning,
                stacklevel=2,
            )
            self.transforms = args[0]
        else:
            self.transforms = list(args)

    def __getitem__(self, item: int | slice) -> BaseTransform:
        if isinstance(item, int):
            return self.transforms[item]

        return MultiTransform(*self.transforms[item])

    def __getstate__(self) -> dict[str, Any]:
        return {'transforms': [(
            transform.__class__.__name__,
            transform.__getstate__(),
        ) for transform in self.transforms]}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.transforms = []

        if isinstance(state['transforms'][0], BaseTransform):
            warn(
                'Transform is saved in old non-weights safe format and is deprecated, '
                'please resave the transform in the new format using net.save()',
                DeprecationWarning,
                stacklevel=2,
            )
            self.transforms = state['transforms']
        else:
            for name, transform in state['transforms']:
                self.transforms.append(globals()[name]())
                self.transforms[-1].__setstate__(transform)

    def forward(self, x: ArrayLike) -> ArrayLike:
        transform: BaseTransform

        for transform in self.transforms:
            x = transform(x)
        return x

    def backward(self, x: ArrayLike) -> ArrayLike:
        transform: BaseTransform

        for transform in self.transforms[::-1]:
            x = transform(x, back=True)
        return x

    def forward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        transform: BaseTransform

        for transform in self.transforms:
            x, uncertainty = transform(x, uncertainty=uncertainty)
        return x, uncertainty

    def backward_grad(
            self,
            x: ArrayLike,
            uncertainty: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        transform: BaseTransform

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

    def extra_repr(self) -> str:
        transform_repr: str
        extra_repr: str = ''

        for i, transform in enumerate(self.transforms):
            transform_repr = repr(transform).replace('\n', '\n\t')
            extra_repr += f"\n\t({i}): {transform_repr},"

        return f'{extra_repr}\n'


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
    extra_repr() -> str
        Displays layer parameters when printing the transform
    """
    @overload
    def __init__(self, *, offset: ndarray, scale: ndarray) -> None:
        ...

    @overload
    def __init__(
            self,
            *,
            data: ArrayLike,
            mean: bool = ...,
            dim: int | tuple[int, ...] | None = ...) -> None:
        ...

    def __init__(
            self,
            mean: bool = True,
            dim: int | tuple[int, ...] | None = None,
            offset: ndarray | None = None,
            scale: ndarray | None = None,
            data: ArrayLike | None = None) -> None:
        """
        Parameters
        ----------
        mean : bool, default = True
            If data should be normalised to zero mean and unit variance, or between 0 and 1
        dim : int | tuple[int, ...] | None, default = None
            Dimensions to normalise over, if None, all dimensions will be normalised over
        offset : ndarray | None, default = None
            Offset to subtract from the data if data argument is None
        scale : ndarray | None, default = None
            Scale to divide the data if data argument is None
        data : ArrayLike | None, default = None
            Data to normalise with shape (N,...), where N is the number of elements
        """
        super().__init__()
        self.offset: ndarray
        self.scale: ndarray

        if isinstance(data, Tensor):
            data = data.cpu().numpy()
        elif data is None:
            self.offset = offset or np.array([0])
            self.scale = scale or np.array([1])
            return

        if mean:
            self.offset = np.mean(data, axis=dim, keepdims=True)
            self.scale = np.std(data, axis=dim, keepdims=True)
        else:
            self.offset = np.amin(data, axis=dim, keepdims=True)
            self.scale = np.amax(data, axis=dim, keepdims=True) - self.offset

    def __getstate__(self) -> dict[str, Any]:
        return {'offset': self.offset.tolist(), 'scale': self.scale.tolist()}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.offset = np.array(state['offset'])
        self.scale = np.array(state['scale'])

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

    def extra_repr(self) -> str:
        if 1 < self.offset.size <= 10:
            return (f"\n\toffset: {np.vectorize(lambda x: f'{x:.2g}')(self.offset)},"
                    f"\n\tscale: {np.vectorize(lambda x: f'{x:.2g}')(self.scale)},\n")

        if self.offset.size > 1:
            return f'offset shape: {self.offset.shape}, scale shape: {self.scale.shape}'

        return f'offset: {self.offset.item():.2g}, scale: {self.scale.item():.2g}'


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
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        """
        Parameters
        ----------
        dtype : dtype, default = float32
            Data type of the tensor
        """
        super().__init__()
        self.dtype: torch.dtype = dtype

    def __getstate__(self) -> dict[str, Any]:
        return {'dtype': self.dtype}

    def __setstate__(self, state: dict[str, Any]) -> None:
        try:
            self.dtype = state['dtype']
        except KeyError:
            self.dtype = torch.float32

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


class Reshape(BaseTransform):
    """
    Reshapes the data

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
    extra_repr() -> str
        Displays layer parameters when printing the transform
    """
    def __init__(
            self,
            in_shape: list[int] | None = None,
            out_shape: list[int] | None = None) -> None:
        """
        Parameters
        ----------
        in_shape : list[int] | None, default = None
            Original shape of the data
        out_shape : list[int] | None, default = None
            Output shape of the data
        """
        super().__init__()
        self._in_shape: list[int] | None = in_shape
        self._out_shape: list[int] | None = out_shape

    def __getstate__(self) -> dict[str, Any]:
        return {'in_shape': self._in_shape, 'out_shape': self._out_shape}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._in_shape = state['in_shape']
        self._out_shape = state['out_shape']

    def forward(self, x: ArrayLike) -> ArrayLike:
        if self._out_shape is None:
            return super().forward(x)
        if isinstance(x, ndarray):
            return x.reshape(len(x), *self._out_shape)
        return x.view(len(x), *self._out_shape)

    def backward(self, x: ArrayLike) -> ArrayLike:
        if self._in_shape is None:
            return super().backward(x)
        if isinstance(x, ndarray):
            return x.reshape(len(x), *self._in_shape)
        return x.view(len(x), *self._in_shape)

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

    def extra_repr(self) -> str:
        return f'in_shape: {self._in_shape}, out_shape: {self._out_shape}'
