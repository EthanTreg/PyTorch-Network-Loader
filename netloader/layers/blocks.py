"""
Composite layer blocks as used in networks such as ConvNeXt
"""
from typing import Any, OrderedDict, cast

from torch import nn, Tensor

from netloader.utils import Shapes
from netloader.utils.types import TensorListLike
from netloader.layers.base import BaseSingleLayer
from netloader.layers.convolutional import Conv, ConvDepth
from netloader.layers.misc import Checkpoint, DropPath, Scale, Shortcut


class ConvNeXtBlock(BaseSingleLayer):
    """
    ConvNeXt block using BaseLayer.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    layers : nn.Sequential
        Layers part of the parent layer to loop through in the forward pass
    shapes : Shapes
        Layer output shapes
    check_shapes : Shapes
        Checkpoint output shapes

    Methods
    -------
    forward(x, outputs, checkpoints) -> Tensor
        Forward pass of the block
    """
    def __init__(
            self,
            net_out: list[int],
            shapes: Shapes,
            *,
            drop_path: float = 0,
            layer_scale: float = 1e-6,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_out : list[int]
            Shape of the network's output, required only if layer contains factor
        shapes : Shapes
            Shape of the outputs from each layer
        drop_path : float, default = 0
            Probability of Drop Path dropout
        layer_scale : float, default = 1e-6
            Initial scale factor for layer Scale
        **kwargs
            Leftover parameters for checking if they are valid
        """
        super().__init__(**kwargs)
        self.shapes: Shapes = shapes[-1:]
        self.check_shapes: Shapes = Shapes([])

        self.layers = nn.Sequential(OrderedDict([
            ('Checkpoint', Checkpoint(self.shapes, self.check_shapes)),
            ('ConvDepth', ConvDepth(
                net_out,
                self.shapes,
                layer=-1,
                factor=1,
                kernel=7,
                stride=1,
                padding=3,
                activation=None,
                norm='layer',
                **kwargs,
            )),
            ('Conv1', Conv(
                net_out,
                self.shapes,
                layer=-1,
                factor=4,
                kernel=1,
                padding='same',
                activation='GELU',
                **kwargs,
            )),
            ('Conv2', Conv(
                net_out,
                self.shapes,
                layer=0,
                factor=1,
                kernel=1,
                padding='same',
                activation=None,
                **kwargs,
            )),
            ('Scale', Scale(1, layer_scale, self.shapes, **kwargs)),
            ('DropPath', DropPath(drop_path, shapes=self.shapes, **kwargs)),
            ('Shortcut', Shortcut(
                True,
                -1,
                self.shapes,
                self.check_shapes,
                **kwargs | {'checkpoint': True},
            )),
        ]))
        shapes.append(self.shapes[-1])

    def forward(
            self,
            x: TensorListLike,
            outputs: list[TensorListLike],
            checkpoints: list[TensorListLike],
            *_: Any,
            **__: Any) -> Tensor:
        """
        Forward pass for the ConvNeXt block

        Parameters
        ----------
        x : TensorListLike
            Input with tensors of shape (N,...) and type float, where N is the batch size
        outputs : list[TensorListLike]
            Output from each layer with tensors of shape (N,...) and type float
        checkpoints : list[TensorListLike]
            List of checkpoint values with tensors of shape (N,...) and type float

        Returns
        -------
        Tensor
            Output tensor with shape (N,...) and type float
        """
        for layer in self.layers:
            x = layer(x, outputs=outputs, checkpoints=checkpoints)
        return cast(Tensor, x)
