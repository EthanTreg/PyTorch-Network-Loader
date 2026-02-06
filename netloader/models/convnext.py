"""
PyTorch-Network-Loader Python implementation of ConvNeXt.
See the ConvNeXt paper: arXiv:2201.03545.
See the ConvNeXt GitHub: https://github.com/facebookresearch/ConvNeXt.
"""
from typing import Any

import numpy as np
from torch import nn

from netloader import layers
from netloader.network import Network
from netloader.utils import Config, Shapes
from netloader.layers.base import BaseLayer


class ConvNeXt(Network):
    """
    ConvNeXt network using Network.

    Original paper implementation: arXiv:2201.03545.

    Attributes
    ----------
    layer_num : int, default = None
        Number of layers to use, if None use all layers
    group : int, default = 0
        Which group is the active group if a layer has the group attribute
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    config : dict[str, Any]
        Network configuration
    kl_loss : Tensor
        KL divergence loss on the latent space, of shape (1) and type float, if using a sample layer
    net : nn.ModuleList
        Network construction
    shapes : Shapes
        Layer output shapes
    check_shapes : Shapes
        Checkpoint output shapes
    """
    def __init__(
            self,
            name: str,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            *,
            max_drop_path: float = 0,
            layer_scale: float = 1e-6,
            dims: list[int] | None = None,
            depths: list[int] | None = None) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the network configuration file
        in_shape : list[int] | tuple[int, ...]
            Shape of the input tensor, excluding batch size
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
            Config(),
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
            Config(),
            state['shapes'][0],
            [],
            suppress_warning=True,
        )
        self._layer_scale = state['layer_scale']
        self._max_drop_path = state['max_drop_path']
        self._dims = state['dims']
        self._depths = state['depths']
        self.shapes = Shapes([state['shapes'][0]])

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

            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def _build_net(self, out_shape: list[int]) -> None:
        """
        Constructs the ConvNeXt network layers.

        Parameters
        ----------
        out_shape : list[int]
            shape of the output tensor, excluding batch size
        """
        assert isinstance(self.shapes, list)
        drop_paths: list[float]
        depths: list[int] = np.cumsum(self._depths).tolist()
        kwargs: dict[str, Any] = {'check_shapes': []}
        drop_paths = np.linspace(0, self._max_drop_path, depths[-1]).tolist()

        # Stem
        self.net.extend(self._stem(out_shape, **kwargs))

        # Main body
        for dim, in_depth, out_depth in zip(self._dims, [0, *depths], depths):
            # Downscaling, this is skipped in the first iteration
            self.net.extend([
                layers.LayerNorm(dims=1, shapes=self.shapes, **kwargs),
                layers.ConvDownscale(
                    out_shape,
                    self.shapes,
                    filters=dim,
                    scale=2,
                    activation=None if self.version != '<3.8.1' else 'ELU',
                    **kwargs,
                ),
            ] if dim != self._dims[0] else [])

            # ConvNeXt blocks
            self.net.extend([
                *[layers.ConvNeXtBlock(
                    out_shape,
                    self.shapes,
                    drop_path=drop_path,
                    layer_scale=self._layer_scale,
                ) for drop_path in drop_paths[in_depth:out_depth]],
            ])

        # Head
        self.net.extend(self._head(out_shape, **kwargs))

    def _head(self, out_shape: list[int], **kwargs: Any) -> list[BaseLayer]:
        """
        Head of ConvNeXt for adapting the learned features into the desired output.

        Parameters
        ----------
        out_shape : list[int]
            shape of the output tensor, excluding batch size

        **kwargs
            Global network parameters

        Returns
        -------
        list[BaseLayer]
            Layers used in the head
        """
        return [
            layers.AdaptivePool(1, self.shapes, **kwargs),
            layers.Reshape([-1], shapes=self.shapes, **kwargs),
            layers.LayerNorm(dims=1, shapes=self.shapes, **kwargs),
            layers.Linear(
                out_shape,
                self.shapes,
                factor=1,
                activation=None if self.version != '<3.8.1' else 'SELU',
                **kwargs,
            ),
        ]

    def _stem(self, out_shape: list[int], **kwargs: Any) -> list[BaseLayer]:
        """
        Stem of ConvNeXt for initial downscaling of the input.

        Parameters
        ----------
        out_shape : list[int]
            shape of the output tensor, excluding batch size

        **kwargs
            Global network parameters

        Returns
        -------
        list[BaseLayer]
            Layers used in the stem
        """
        return [
            layers.ConvDownscale(
                out_shape,
                self.shapes,
                filters=self._dims[0],
                scale=4,
                activation=None if self.version != '<3.8.1' else 'ELU',
                norm='layer',
                **kwargs,
            ),
        ]


class ConvNeXtTiny(ConvNeXt):
    """
    Tiny version of ConvNeXt with the number of convolutional filters in each downscaled section
    following [96, 192, 384, 768] with the number of ConvNeXt blocks in each downscaled section
    following [3, 3, 9, 3].

    Original paper implementation: arXiv:2201.03545.

    Attributes
    ----------
    layer_num : int, default = None
        Number of layers to use, if None use all layers
    group : int, default = 0
        Which group is the active group if a layer has the group attribute
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    config : dict[str, Any]
        Network configuration
    kl_loss : Tensor
        KL divergence loss on the latent space, of shape (1) and type float, if using a sample layer
    net : nn.ModuleList
        Network construction
    shapes : Shapes
        Layer output shapes
    check_shapes : Shapes
        Checkpoint output shapes
    """
    def __init__(
            self,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            *,
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
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
            'convnext_tiny',
            in_shape,
            out_shape,
            max_drop_path=max_drop_path,
            layer_scale=layer_scale,
        )


class ConvNeXtSmall(ConvNeXt):
    """
    Small version of ConvNeXt with the number of convolutional filters in each downscaled section
    following [96, 192, 384, 768] with the number of ConvNeXt blocks in each downscaled section
    following [3, 3, 27, 3].

    Original paper implementation: arXiv:2201.03545.

    Attributes
    ----------
    layer_num : int, default = None
        Number of layers to use, if None use all layers
    group : int, default = 0
        Which group is the active group if a layer has the group attribute
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    config : dict[str, Any]
        Network configuration
    kl_loss : Tensor
        KL divergence loss on the latent space, of shape (1) and type float, if using a sample layer
    net : nn.ModuleList
        Network construction
    shapes : Shapes
        Layer output shapes
    check_shapes : Shapes
        Checkpoint output shapes
    """
    def __init__(
            self,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            *,
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
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
            'convnext_small',
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
    following [3, 3, 27, 3].

    Original paper implementation: arXiv:2201.03545.

    Attributes
    ----------
    layer_num : int, default = None
        Number of layers to use, if None use all layers
    group : int, default = 0
        Which group is the active group if a layer has the group attribute
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    config : dict[str, Any]
        Network configuration
    kl_loss : Tensor
        KL divergence loss on the latent space, of shape (1) and type float, if using a sample layer
    net : nn.ModuleList
        Network construction
    shapes : Shapes
        Layer output shapes
    check_shapes : Shapes
        Checkpoint output shapes
    """
    def __init__(
            self,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            *,
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
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
            'convnext_base',
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
    following [3, 3, 27, 3].

    Original paper implementation: arXiv:2201.03545.

    Attributes
    ----------
    layer_num : int, default = None
        Number of layers to use, if None use all layers
    group : int, default = 0
        Which group is the active group if a layer has the group attribute
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    config : dict[str, Any]
        Network configuration
    kl_loss : Tensor
        KL divergence loss on the latent space, of shape (1) and type float, if using a sample layer
    net : nn.ModuleList
        Network construction
    shapes : Shapes
        Layer output shapes
    check_shapes : Shapes
        Checkpoint output shapes
    """
    def __init__(
            self,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            *,
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
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
            'convnext_large',
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
    section following [3, 3, 27, 3].

    Original paper implementation: arXiv:2201.03545.

    Attributes
    ----------
    layer_num : int, default = None
        Number of layers to use, if None use all layers
    group : int, default = 0
        Which group is the active group if a layer has the group attribute
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    config : dict[str, Any]
        Network configuration
    kl_loss : Tensor
        KL divergence loss on the latent space, of shape (1) and type float, if using a sample layer
    net : nn.ModuleList
        Network construction
    shapes : Shapes
        Layer output shapes
    check_shapes : Shapes
        Checkpoint output shapes
    """
    def __init__(
            self,
            in_shape: list[int] | tuple[int, ...],
            out_shape: list[int],
            *,
            max_drop_path: float = 0,
            layer_scale: float = 1e-6) -> None:
        """
        Parameters
        ----------
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
            'convnext_xlarge',
            in_shape,
            out_shape,
            max_drop_path=max_drop_path,
            layer_scale=layer_scale,
            dims=[256, 512, 1024, 2048],
            depths=[3, 3, 27, 3],
        )
