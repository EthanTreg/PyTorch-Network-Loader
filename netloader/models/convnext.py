"""
PyTorch-Network-Loader Python implementation of ConvNeXt.
See the ConvNeXt paper: arXiv:2201.03545.
See the ConvNeXt GitHub: https://github.com/facebookresearch/ConvNeXt
"""
from typing import Any, OrderedDict

import numpy as np
from torch import nn, Tensor
from netloader import layers
from netloader.network import Network
from netloader.layers.base import BaseLayer


class ConvNeXtBlock(BaseLayer):
    """
    ConvNeXt block using BaseLayer.

    Attributes
    ----------
    group : int, default = 0
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    shapes : list[list[int]]
        Layer output shapes
    check_shapes : list[list[int]]
        Checkpoint output shapes
    layers : Sequential
        Layers part of the parent layer to loop through in the forward pass

    Methods
    -------
    forward(x) -> Tensor
        Forward pass of the block
    """
    def __init__(
            self,
            net_out: list[int],
            shapes: list[list[int]],
            drop_path: float = 0,
            layer_scale: float = 1e-6,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_out : list[int]
            Shape of the network's output, required only if layer contains factor
        shapes : list[list[int]]
            Shape of the outputs from each layer
        drop_path : float, default = 0
            Probability of Drop Path dropout
        layer_scale : float, default = 1e-6
            Initial scale factor for layer Scale

        **kwargs
            Leftover parameters for checking if they are valid
        """
        kwargs = {'idx': 0} | kwargs
        super().__init__(**kwargs)
        self.shapes: list[list[int]] = [shapes[-1].copy()]
        self.check_shapes: list[list[int]] = []

        self.layers = nn.Sequential(OrderedDict([
            ('Checkpoint', layers.Checkpoint(self.shapes, self.check_shapes)),
            ('ConvDepth', layers.ConvDepth(
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
            ('Conv1', layers.Conv(
                net_out,
                self.shapes,
                layer=-1,
                factor=4,
                kernel=1,
                padding='same',
                activation='GELU',
                **kwargs,
            )),
            ('Conv2', layers.Conv(
                net_out,
                self.shapes,
                layer=0,
                factor=1,
                kernel=1,
                padding='same',
                activation=None,
                **kwargs,
            )),
            ('Scale', layers.Scale(1, layer_scale, self.shapes, **kwargs)),
            ('DropPath', layers.DropPath(
                drop_path,
                self.shapes,
                **kwargs,
            )),
            ('Shortcut', layers.Shortcut(
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
            x: Tensor,
            outputs: list[Tensor],
            checkpoints: list[Tensor],
            *_: Any,
            **__: Any) -> Tensor:
        """
        Forward pass for the ConvNeXt block

        Parameters
        ----------
        x : (N,...) Tensor
            Input tensor with batch size N
        outputs : list[Tensor]
            Output from each layer
        checkpoints : list[Tensor]
            List of checkpoint values

        Returns
        -------
        (N,...) Tensor
            Output tensor with batch size N
        """
        for layer in self.layers:
            x = layer(x, outputs=outputs, checkpoints=checkpoints)
        return x


class ConvNeXt(Network):
    """
    ConvNeXt network using Network.

    Attributes
    ----------
    name : str
        Name of the network, used for saving
    check_shapes : list[list[int]]
        Checkpoint output shapes
    shapes : list[list[int] | list[list[int]]
        Layer output shapes
    checkpoints : list[Tensor]
        Outputs from each checkpoint
    config : dict[str, Any]
        Network configuration
    net : ModuleList
        Network construction
    layer_num : int, default = None
        Number of layers to use, if None use all layers
    group : int, default = 0
        Which group is the active group if a layer has the group attribute
    kl_loss : Tensor, default = 0
        KL divergence loss on the latent space, if using a sample layer
    """
    def __init__(
            self,
            name: str,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            max_drop_path: float = 0,
            layer_scale: float = 1e-6,
            dims: list[int] | None = None,
            depths: list[int] | None = None) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the network configuration file
        in_shape : list[int] | list[list[int]] | tuple[int, ...]
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        max_drop_path : float, default = 0
            Maximum drop path fraction
        layer_scale : float, default = 1e-6
            Default value for learnable parameters to scale the convolutional filters in a ConvNeXt
            block
        dims : list[int], default = [96, 192, 384, 768]
            Number of convolutional filters in each downscaled section of the network
        depths : list[int], default = [3, 3, 9, 3]
            Number of ConvNeXt blocks in each downscaled section of the network
        """
        super().__init__(
            name,
            {'net': {}, 'layers': []},
            in_shape,
            [],
            suppress_warning=True,
        )
        self._layer_scale: float = layer_scale
        self._max_drop_path: float = max_drop_path
        self._dims: list[int] = dims or  [96, 192, 384, 768]
        self._depths: list[int] = depths or [3, 3, 9, 3]
        self._build_net(out_shape)
        self.apply(self._init_weights)

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the network for pickling

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the network
        """
        return super().__getstate__() | {
            'dims': self._dims,
            'depths': self._depths,
            'layer_scale': self._layer_scale,
            'max_drop_path': self._max_drop_path,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the network for pickling

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the network
        """
        super().__init__(
            state['name'],
            {'net': {}, 'layers': []},
            state['shapes'][0],
            [],
            suppress_warning=True,
        )
        self._layer_scale = state['layer_scale']
        self._max_drop_path = state['max_drop_path']
        self._dims = state['dims']
        self._depths = state['depths']
        self.group = state['group']
        self.layer_num = state['layer_num']
        self.shapes = [state['shapes'][0]]

        self._build_net(state['shapes'][-1])
        self.load_state_dict(state['net'])

    @staticmethod
    def _init_weights(layer: nn.Module) -> None:
        """
        Initialises the weights of the convolutional and linear layers.

        Parameters
        ----------
        layer : nn.Module
            Layer to initialise if it is convolutional or linear
        """
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias, 0)

    def _build_net(self, out_shape: list[int]) -> None:
        """
        Constructs the ConvNeXt network layers

        Parameters
        ----------
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        """
        assert isinstance(self.shapes, list)
        drop_paths: list[float]
        depths: list[int] = np.cumsum(self._depths).tolist()
        kwargs: dict[str, Any] = {'idx': 0, 'check_shapes': []}
        drop_paths = np.linspace(0, self._max_drop_path, depths[-1]).tolist()

        # Stem
        self.net.append(layers.Conv(
            out_shape,
            self.shapes,
            filters=self._dims[0],
            kernel=4,
            stride=4,
            norm='layer',
            **kwargs,
        ))

        # Main body
        for dim, in_depth, out_depth in zip(self._dims, [0, *depths], depths):
            # Downscaling
            self.net.extend([
                layers.LayerNorm(dims=1, shapes=self.shapes, **kwargs),
                layers.Conv(
                    out_shape,
                    self.shapes,
                    filters=dim,
                    kernel=2,
                    stride=2,
                    **kwargs,
                ),
            ] if dim != self._dims[0] else [])

            # ConvNeXt blocks
            self.net.extend([
                *[ConvNeXtBlock(
                    out_shape,
                    self.shapes,
                    drop_path,
                    layer_scale=self._layer_scale,
                ) for drop_path in drop_paths[in_depth:out_depth]],
            ])

        # Head
        self.net.extend([
            layers.AdaptivePool(1, self.shapes, **kwargs),
            layers.Reshape([-1], shapes=self.shapes, **kwargs),
            layers.LayerNorm(dims=1, shapes=self.shapes, **kwargs),
            layers.Linear(out_shape, self.shapes, factor=1, **kwargs),
        ])


class ConvNeXtTiny(ConvNeXt):
    """
    Tiny version of ConvNeXt with the number of convolutional filters in each downscaled section
    following [96, 192, 384, 768] with the number of ConvNeXt blocks in each downscaled section
    following [3, 3, 9, 3]
    """
    def __init__(
            self,
            name: str,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the network configuration file
        in_shape : list[int] | list[list[int]] | tuple[int, ...]
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        max_drop_path : float, default = 0
            Maximum drop path fraction
        layer_scale : float, default = 1e-6
            Default value for learnable parameters to scale the convolutional filters in a ConvNeXt
            block
        """
        super().__init__(
            name,
            in_shape,
            out_shape,
            max_drop_path=max_drop_path,
            layer_scale=layer_scale,
        )


class ConvNeXtSmall(ConvNeXt):
    """
    Small version of ConvNeXt with the number of convolutional filters in each downscaled section
    following [96, 192, 384, 768] with the number of ConvNeXt blocks in each downscaled section
    following [3, 3, 27, 3]
    """
    def __init__(
            self,
            name: str,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the network configuration file
        in_shape : list[int] | list[list[int]] | tuple[int, ...]
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        max_drop_path : float, default = 0
            Maximum drop path fraction
        layer_scale : float, default = 1e-6
            Default value for learnable parameters to scale the convolutional filters in a ConvNeXt
            block
        """
        super().__init__(
            name,
            in_shape,
            out_shape,
            max_drop_path=max_drop_path,
            layer_scale=layer_scale,
            depths=[3, 3, 27, 3],
        )


class ConvNeXtBase(ConvNeXt):
    """
    Base version of ConvNeXt with the number of convolutional filters in each downscaled section
    following [128, 256, 512, 1024] with the number of ConvNeXt blocks in each downscaled section
    following [3, 3, 27, 3]
    """
    def __init__(
            self,
            name: str,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the network configuration file
        in_shape : list[int] | list[list[int]] | tuple[int, ...]
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        max_drop_path : float, default = 0
            Maximum drop path fraction
        layer_scale : float, default = 1e-6
            Default value for learnable parameters to scale the convolutional filters in a ConvNeXt
            block
        """
        super().__init__(
            name,
            in_shape,
            out_shape,
            max_drop_path=max_drop_path,
            layer_scale=layer_scale,
            dims=[128, 256, 512, 1024],
            depths=[3, 3, 27, 3],
        )


class ConvNeXtLarge(ConvNeXt):
    """
    Large version of ConvNeXt with the number of convolutional filters in each downscaled section
    following [192, 384, 768, 1536] with the number of ConvNeXt blocks in each downscaled section
    following [3, 3, 27, 3]
    """
    def __init__(
            self,
            name: str,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the network configuration file
        in_shape : list[int] | list[list[int]] | tuple[int, ...]
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        max_drop_path : float, default = 0
            Maximum drop path fraction
        layer_scale : float, default = 1e-6
            Default value for learnable parameters to scale the convolutional filters in a ConvNeXt
            block
        """
        super().__init__(
            name,
            in_shape,
            out_shape,
            max_drop_path=max_drop_path,
            layer_scale=layer_scale,
            dims=[192, 384, 768, 1536],
            depths=[3, 3, 27, 3],
        )


class ConvNeXtXLarge(ConvNeXt):
    """
    Extra large version of ConvNeXt with the number of convolutional filters in each downscaled
    section following [256, 512, 1024, 2048] with the number of ConvNeXt blocks in each downscaled
    section following [3, 3, 27, 3]
    """
    def __init__(
            self,
            name: str,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        super().__init__(
            name,
            in_shape,
            out_shape,
            max_drop_path=max_drop_path,
            layer_scale=layer_scale,
            dims=[256, 512, 1024, 2048],
            depths=[3, 3, 27, 3],
        )
