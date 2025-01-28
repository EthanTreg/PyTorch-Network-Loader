"""
Layer utility functions
"""
from warnings import warn

# For compatibility
from netloader.layers import base


class BaseLayer(base.BaseLayer):
    """
    Deprecated BaseLayer class as BaseLayer was moved to netloader.layers.base
    """
    def __init_subclass__(cls, **kwargs):
        warn(
            'BaseLayer is deprecated, use BaseLayer from netloader.layers.base instead',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        warn(
            'BaseLayer is deprecated, use BaseLayer from netloader.layers.base instead',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class BaseMultiLayer(base.BaseMultiLayer):
    """
    Deprecated MultiBaseLayer class as MultiBaseLayer was moved to netloader.layers.base
    """
    def __init_subclass__(cls, **kwargs):
        warn(
            'BaseMultiLayer is deprecated, use BaseLayer from netloader.layers.base',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)

    def __init__(self, *args, **kwargs):
        warn(
            'BaseMultiLayer is deprecated, use BaseLayer from netloader.layers.base '
            'instead',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def _int_list_conversion(length: int, elements: list[int | list[int]]) -> list[list[int]]:
    """
    Converts integers to a list of integers, if integer is already a list of integers, then list
    will be preserved

    Parameters
    ----------
    length : int
        Number of elements in the converted list of integers
    elements : list[int | list[int]]
        Integers to convert to list of integers, or list of integers to remain unchanged

    Returns
    -------
    list[list[int]]
        List of integers for each inputted integer or list of integers
    """
    lists: list[list[int]] = []

    for element in elements:
        if isinstance(element, int):
            lists.append([element] * length)
        else:
            lists.append(element)

    return lists


def _kernel_shape(
        kernel: int | list[int],
        strides: int | list[int],
        padding: int | list[int],
        shape: list[int]) -> list[int]:
    """
    Calculates the output shape after a kernel operation

    Parameters
    ----------
    kernel : int | list[int]
        Size of the kernel
    strides : int | list[int]
        Stride of the kernel
    padding : int | list[int]
        Input padding
    shape : list[int]
        Input shape of the layer

    Returns
    -------
    list[int]
        Output shape of the layer
    """
    shape = shape.copy()
    strides, kernel, padding = _int_list_conversion(
        len(shape[1:]),
        [strides, kernel, padding],
    )

    for i, (stride, kernel_length, pad, length) in enumerate(zip(
            strides,
            kernel,
            padding,
            shape[1:]
    )):
        shape[i + 1] = max(1, (length + 2 * pad - kernel_length) // stride + 1)

    return shape


def _padding(
        kernel: int | list[int],
        strides: int | list[int],
        in_shape: list[int],
        out_shape: list[int]) -> list[int]:
    """
    Calculates the padding required for specific output shape

    Parameters
    ----------
    kernel : int | list[int]
        Size of the kernel
    strides : int | list[int]
        Stride of the kernel
    in_shape : list[int]
        Input shape of the layer
    out_shape : list[int]
        Output shape from the layer

    Returns
    -------
    list[int]
        Required padding for specific output shape
    """
    padding: list[int] = []
    strides, kernel = _int_list_conversion(len(in_shape[1:]), [strides, kernel])

    for stride, kernel_length, in_length, out_length in zip(
            strides,
            kernel,
            in_shape[1:],
            out_shape[1:],
    ):
        padding.append((stride * (out_length - 1) + kernel_length - in_length) // 2)

    return padding
