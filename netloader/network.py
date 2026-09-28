"""
Constructs a network from layers and can load weights to resume network training
"""
from __future__ import annotations
import os
import json
import logging as log
from warnings import warn
from contextlib import nullcontext
from abc import ABC, abstractmethod
from typing import (
    Any,
    Literal,
    TextIO,
    Self,
    Iterator,
    Generic,
    cast,
    overload,
    TYPE_CHECKING,
)

import torch
from torch import nn, Tensor

import netloader
from netloader import layers, utils
from netloader.utils.types import TensorListLike, ModuleListT
from netloader.utils import configs, Config, Shapes, TypedModuleList

if TYPE_CHECKING:
    from zuko.distributions import NormalizingFlow


class BaseNetwork(nn.Module, ABC, Generic[ModuleListT]):
    """
    Base class for all networks in NetLoader

    Attributes
    ----------
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    kl_loss : Tensor
        KL divergence loss on the latent space of shape (1) and type float, if using a sample layer
    layers : ModuleListT
        Network construction
    """
    _logger: log.Logger = log.getLogger(__name__)
    name: str
    version: str = netloader.__version__
    checkpoints: list[TensorListLike]
    kl_loss: Tensor
    layers: ModuleListT

    def __init__(self) -> None:
        super().__init__()
        self.checkpoints = []
        self.kl_loss = torch.tensor(0.)

        # Adds Network class to list of safe PyTorch classes when loading saved networks
        torch.serialization.add_safe_globals([self.__class__])

    def __getattr__(self, item: str) -> Any:
        """
        Returns the attribute of the network.

        Parameters
        ----------
        item : str
            Name of the attribute to return

        Returns
        -------
        Any
            Attribute of the network
        """
        if item == 'net':
            warn(
                'BaseNetwork.net is deprecated, please use BaseNetwork.layers '
                'instead',
                DeprecationWarning,
                stacklevel=2,
            )
            return self.layers
        return super().__getattr__(item)

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the network for pickling

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the network
        """
        return {'name': self.name, 'version': netloader.__version__}

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Sets the attribute of the network.

        Parameters
        ----------
        key : str
            Name of the attribute to set
        value : Any
            Value to set the attribute to
        """
        if key == 'net':
            warn(
                'BaseNetwork.net is deprecated, please use BaseNetwork.layers '
                'instead',
                DeprecationWarning,
                stacklevel=2,
            )
            super().__setattr__('layers', value)
        else:
            super().__setattr__(key, value)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the network for pickling

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the network
        """
        super().__init__()
        self.name = state['name']
        self.version = state.get('version', '<3.9.0')
        self.checkpoints = []
        self.kl_loss = torch.tensor(0.)

        if not utils.compare_versions(self.version, netloader.__version__, level='minor'):
            warn(
                f'Network version ({self.version}) is older than the current NetLoader '
                f'version ({netloader.__version__}), please resave the network',
                DeprecationWarning,
                stacklevel=2,
            )

    def _load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Loads the state dictionary of the network.
        If the keys in the state dictionary do not match the keys in the network, it will attempt to
        match the weights without using the keys.

        Parameters
        ----------
        state_dict : dict[str, Any]
            State dictionary of the network
        """
        try:
            self.load_state_dict(state_dict)
        except RuntimeError:
            log.getLogger().error('Failed to load network weights due to incorrect weight names, '
                                    'will try to match weights.')
            self.load_state_dict(dict(zip(self.state_dict(), state_dict.values())))

    @abstractmethod
    def forward(self, x: TensorListLike) -> TensorListLike:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : TensorListLike
            Input tensor with shape (N,...) and type float, where N is the batch size

        Returns
        -------
        TensorListLike
            Output tensor from the network with shape (N,...) and type float
        """

    def clear_checkpoints(self) -> None:
        """
        Clears the checkpoints of the network.
        """
        self.checkpoints = []

    @overload
    def get_config(self, dict_: Literal[True] = ...) -> dict[str, Any]: ...

    @overload
    def get_config(self, dict_: Literal[False] = ...) -> Config: ...

    def get_config(self, dict_: bool = True) -> dict[str, Any] | Config:
        """
        Returns the network configuration.

        Parameters
        ----------
        dict_ : bool, Optional
            If configuration should be returned as a dictionary, default = True

        Returns
        -------
        dict[str, Any] | Config
            Network configuration
        """
        return Config().to_dict() if dict_ else Config()

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self.kl_loss = self.kl_loss.detach().to(*args, **kwargs)

        for i, checkpoint in enumerate(self.checkpoints):
            if hasattr(checkpoint, 'to'):
                self.checkpoints[i] = checkpoint.detach().to(*args, **kwargs)
            else:
                self._logger.warning(f'Failed to move checkpoint {checkpoint.__class__.__name__} '
                                     f'to device')
        return self


class CompatibleNetwork(BaseNetwork[nn.Module | nn.ModuleList]):
    """
    A wrapper for Module that ensures compatibility with BaseArchitecture by adding required
    attributes.

    Attributes
    ----------
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    kl_loss : Tensor
        KL divergence loss on the latent space of shape (1) and type float, if using a sample layer
    layers : Module | ModuleList
        Network construction
    """
    def __init__(self, net: nn.Module | nn.ModuleList, *, name: str = '') -> None:
        """
        Parameters
        ----------
        net : Module | ModuleList
            The neural network module to wrap or list of layers in the network
        name : str, Optional
            Name of the network, used for saving
        """
        super().__init__()
        self.layers = net

        if name:
            self.name = name
        elif hasattr(self.layers, 'name') and isinstance(self.layers.name, str):
            self.name = self.layers.name
        elif isinstance(self.layers, nn.ModuleList):
            self.name = self.__class__.__name__
        else:
            self.name = self.layers.__class__.__name__

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'layers': [*self.layers] if isinstance(self.layers, nn.ModuleList) else self.layers,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        layers_: list[nn.Module] | nn.Module = cast(
            list[nn.Module] | nn.Module,
            state.get('layers', state.get('net')),
        )

        if isinstance(layers_, list):
            self.layers = nn.ModuleList(layers_)
        elif isinstance(layers_, nn.Module):
            self.layers = layers_
        else:
            self.load_state_dict(layers_)  # type: ignore[unreachable]

    def clear_checkpoints(self) -> None:
        """
        Clears the checkpoints of the network.
        """
        super().clear_checkpoints()
        layer: nn.Module

        if not isinstance(self.layers, nn.ModuleList):
            return

        for layer in self.layers:
            if isinstance(layer, layers.BaseLayer):
                layer.clear_checkpoints()

    def forward(self, x: TensorListLike) -> TensorListLike:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : TensorListLike
            Input tensor with shape (N,...) and type float, where N is the batch size

        Returns
        -------
        TensorListLike
            Output tensor from the network with shape (N,...) and type float
        """
        layer: nn.Module

        if not isinstance(self.layers, nn.ModuleList):
            return self.layers(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self.layers.to(*args, **kwargs)
        return self


class Network(BaseNetwork[TypedModuleList[layers.BaseLayer]]):
    """
    Constructs a neural network from a configuration file.

    Attributes
    ----------
    group : int
        Which group is the active group if a layer has the group attribute
    layer_num : int | None
        Number of layers to use, if None use all layers
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    kl_loss : Tensor
        KL divergence loss on the latent space of shape (1) and type float, if using a sample layer
    layers : TypedModuleList[BaseLayer]
        Network layers
    shapes : Shapes
        Layer output shapes
    check_shapes : Shapes
        Checkpoint output shapes
    """
    _checkpoints: bool
    _config: Config
    group: int = 0
    layer_num: int | None = None
    shapes: Shapes
    check_shapes: Shapes

    def __init__(
            self,
            name: str,
            config: str | Config,
            in_shape: list[int] | list[list[int]] | tuple[int, ...],
            out_shape: list[int],
            *,
            suppress_warning: bool = False,
            root: str = '',
            defaults: dict[str, Any] | None = None) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the network configuration file
        config : str | Config
            Path to the network config directory or configuration dictionary
        in_shape : list[int] | list[list[int]] | tuple[int, ...]
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            Shape of the output tensor, excluding batch size
        suppress_warning : bool, Optional
            If output shape mismatch warning should be suppressed, default = False
        root : str, Optional
            Root directory to prepend to config file path
        defaults : dict[str, Any] | None, Optional
            Default values for the parameters for each type of layer
        """
        super().__init__()
        layer: layers.BaseLayer
        self.name = name.replace('.json', '')

        # Construct layers in network
        self.layers, self.shapes, self.check_shapes, self._config = _create_network(
            os.path.join(config, self.name) if isinstance(config, str) else config,
            in_shape if isinstance(in_shape, list) else list(in_shape),
            list(out_shape),
            suppress_warning=suppress_warning,
            root=root,
            defaults=defaults,
        )
        self._checkpoints = self._config.net.checkpoints
        self._config.layers = [layer.__getstate__() for layer in self.layers]

    def __getattr__(self, item: str) -> Any:
        """
        Returns the attribute of the network.

        Parameters
        ----------
        item : str
            Name of the attribute to return

        Returns
        -------
        Any
            Attribute of the network
        """
        if item == 'config':
            warn(
                'Network.config is deprecated, please use Network.get_config() instead',
                DeprecationWarning,
                stacklevel=2,
            )
            return self.get_config(True)
        return super().__getattr__(item)

    @overload
    def __getitem__(self, idx: int) -> layers.BaseLayer: ...

    @overload
    def __getitem__(self, idx: slice) -> TypedModuleList[layers.BaseLayer]: ...

    def __getitem__(self, idx: int | slice) -> nn.Module | TypedModuleList[layers.BaseLayer]:
        """
        Returns the layer at the specified index.

        Parameters
        ----------
        idx : int | slice
            Index or slice of the layer(s) to return

        Returns
        -------
        Module | TypedModuleList[BaseLayer]
            Layer(s) at the specified index or slice
        """
        return self.layers[idx]

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'group': self.group,
            'layer_num': self.layer_num,
            'shapes': list(self.shapes),
            'config': self.get_config(True),
            'layers': self.state_dict(),
        }

    def __iter__(self) -> Iterator[layers.BaseLayer]:
        """
        Returns an iterator over the layers in the network.

        Returns
        -------
        Iterator[Module]
            Iterator over the layers in the network
        """
        i: int

        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """
        Returns the number of layers in the network.

        Returns
        -------
        int
            Number of layers in the network
        """
        return len(self.layers[:self.layer_num])

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        shapes = Shapes(state['shapes'])
        self.group = state['group']
        self.layer_num = state['layer_num']

        if 'config' not in state:
            warn(
                'Network is saved in old deprecated format and net key in network JSON '
                'file is lost, please resave the network and update Network._config with the JSON '
                'file',
                DeprecationWarning,
                stacklevel=2,
            )
            self._checkpoints = state['_checkpoints']
            self.shapes = Shapes(state['shapes'])
            self.check_shapes = Shapes(state['check_shapes'])
            self._config = Config.from_dict({
                'net': {'checkpoints': state['checkpoints']},
                'layers': state['layers'],
            })
            self.layers = state['_modules']
            return

        self.layers, self.shapes, self.check_shapes, self._config = _create_network(
            Config.from_dict(state['config']),
            shapes.get(0, True),
            shapes[-1] if shapes.check(-1) else shapes[(-1, 0)],
            suppress_warning=True,
        )
        self._checkpoints = self._config.net.checkpoints
        self._load_state_dict(cast(dict[str, Any], state.get('layers', state.get('net'))))

    def clear_checkpoints(self) -> None:
        """
        Clears the checkpoints of the network.
        """
        super().clear_checkpoints()
        layer: layers.BaseLayer

        for layer in self.layers:
            layer.clear_checkpoints()

    def forward(self, x: TensorListLike) -> NormalizingFlow | TensorListLike:
        """
        Forward pass of the network.

        Parameters
        ----------
        x : TensorListLike
            Input tensor(s) with shape (N,...) and type float, where N is the
            batch size

        Returns
        -------
        NormalizingFlow | TensorListLike
            Output tensor from the network with shape (N,...) and type float, or NormalizingFlow if
            the last layer is from layers.flows
        """
        i: int
        outputs: list[TensorListLike] = []
        e: Exception
        layer: layers.BaseLayer
        self.checkpoints = []

        if not self._checkpoints:
            outputs = [x]

        for i, layer in enumerate(self.layers[:self.layer_num]):
            if layer.group not in (0, self.group):
                if not self._checkpoints:
                    outputs.append(torch.tensor([]))

                if isinstance(layer, layers.Checkpoint):
                    self.checkpoints.append(torch.tensor([]))
                continue

            try:
                x = layer(x, outputs=outputs, checkpoints=self.checkpoints, net=self)
            except Exception as e:
                raise type(e)(f'{e}\nError in {layer.__class__.__name__} (layer {i})') from e

            if not self._checkpoints:
                outputs.append(x)
        return x

    @overload
    def get_config(self, dict_: Literal[True] = ...) -> dict[str, Any]: ...

    @overload
    def get_config(self, dict_: Literal[False] = ...) -> Config: ...

    def get_config(self, dict_: bool = True) -> dict[str, Any] | Config:
        return self._config.to_dict() if dict_ else self._config

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        layer: layers.BaseLayer

        for layer in self.layers:
            layer.to(*args, **kwargs)
        return self


class Branch(layers.BaseLayer):
    """
    Creates a branching layer that processes the input through multiple branches and combines the
    outputs.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    branches : list[BaseLayer]
        List of subnetworks for each branch
    head : Concatenate | Pack | Shortcut
        Layer to combine the outputs from each branch, either concatenation, packing or summation
    """
    def __init__(
            self,
            net_check: bool,
            branches: list[list[dict[str, Any]]],
            shapes: Shapes,
            *,
            cache: bool = False,
            checkpoint: bool = True,
            inherit_defaults: bool = True,
            dim: int = 0,
            channels: int | None = None,
            root: str = '',
            method: Literal['concat', 'pack', 'sum'] = 'concat',
            shape: list[int] | None = None,
            defaults: dict[str, Any] | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        branches : list[list[dict[str, Any]]]
            List of layer configurations for each branch
        shapes : Shapes
            Shape of the outputs from each layer
        cache : bool, Optional
            If checkpoints should be saved or cleared after forward pass, default = False
        checkpoint : bool, Optional
            If layer index should be relative to checkpoint layers, default = True
        inherit_defaults : bool, Optional
            If default values for each type of layer should be inherited from the parent network,
            default = True
        dim : int, Optional
            Dimension to concatenate along if method is 'concat', default = 0
        channels : int, Optional
            Number of output channels, won't be used if shape is provided, if channels and
            shape aren't provided, the input dimensions will be preserved
        root : str, Optional
            Root directory to prepend to config file path
        method : Literal['concat', 'pack', 'sum'], Optional
            Method to combine the outputs from each branch, either concatenation, packing or
            summation, default = 'concat'
        shape : list[int] | None, Optional
            Output shape of the block, will be used if provided; otherwise, channels will be used
        defaults : dict[str, Any] | None, Optional
            Default values for the parameters for each type of layer
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._cache: bool = cache
        self._checkpoint: bool = checkpoint or net_check
        self._inherit_defaults: bool = inherit_defaults
        self._dim: int = dim
        self._target_shape: list[int]
        self._method: Literal['concat', 'pack', 'sum'] = method
        self.branches: TypedModuleList[Network] = TypedModuleList()
        self.head: layers.Concatenate | layers.Pack | layers.Shortcut
        in_shape: list[int] | list[list[int]] = shapes.get(-1, True)
        out_shapes: list[list[int]] = []
        kwargs: dict[str, Any]
        net_config: configs.NetConfig = configs.NetConfig(checkpoints=self._checkpoint)
        defaults = defaults or {}

        # Subnetwork output if provided
        if shape:
            self._target_shape = shape
        elif channels:
            self._target_shape = shapes[-1]
            self._target_shape[0] = channels
        else:
            self._target_shape = shapes[-1].copy()

        for i, branch in enumerate(branches):
            self.branches.append(Network(
                f'Branch_{i}',
                Config(layers=branch, net=net_config),
                in_shape,
                self._target_shape,
                suppress_warning=True,
                root=root,
                defaults=defaults | {'checkpoints': self._checkpoint},
            ))
            out_shapes.append(self.branches[-1].shapes[-1])

        kwargs = {
            'net_check': False,
            'layer': list(range(len(out_shapes) - 1)),
            'shapes': Shapes(out_shapes),
            'check_shapes': Shapes(),
        }

        if self._method == 'concat':
            self.head = layers.Concatenate(dim=self._dim, **kwargs)
        elif self._method == 'pack':
            self.head = layers.Pack(**kwargs)
        elif self._method == 'sum':
            self.head = layers.Shortcut(**kwargs)
        else:
            raise ValueError(f'Unknown output combination method {self._method}, must be one of '
                             f'"concat", "pack", or "sum"')

        shapes.append(kwargs['shapes'].get(-1, True))

    def __getstate__(self) -> dict[str, Any]:
        branch: Network
        return super().__getstate__() | {
            'branches': [branch.get_config(True)['layers'] for branch in self.branches],
            'checkpoint': self._checkpoint,
            'inherit_defaults': self._inherit_defaults,
            'dim': self._dim,
            'method': self._method,
            'shape': self._target_shape,
        }

    def clear_checkpoints(self) -> None:
        branch: Network

        for branch in self.branches:
            branch.clear_checkpoints()

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return (f'checkpoint={self._checkpoint}, dim={self._dim}, method={self._method}, '
                f'shape={self._target_shape}')

    def forward(self, x: TensorListLike, *_: Any, **__: Any) -> TensorListLike:
        """
        Forward pass for the Branch layer.

        Parameters
        ----------
        x : TensorListT
            Input tensor(s) with dtype float32 and shape (N,...), where N is batch size

        Returns
        -------
        TensorListT
            Output tensor(s) from the subnetwork with dtype float32 and shape (N,...)
        """
        outputs: list[TensorListLike] = []
        branch: Network

        for branch in self.branches:
            outputs.append(branch(x.clone()))

        if not self._cache:
            self.clear_checkpoints()
        return self.head(outputs[-1], outputs=outputs, checkpoints=[])

    def to(self, *args: Any, **kwargs: Any) -> Self:
        branch: Network
        self.head.to(*args, **kwargs)

        for branch in self.branches:
            branch.to(*args, **kwargs)
        return self


class Composite(layers.BaseLayer):
    """
    Creates a subnetwork from a configuration file.

    Attributes
    ----------
    group : int
        Layer group, if 0 it will always be used, else it will only be used if its group matches the
        Networks
    description : str
        Description of the layer
    net : Network
        Subnetwork for the Composite layer
    """
    def __init__(
            self,
            net_check: bool,
            name: str,
            shapes: Shapes,
            *,
            cache: bool = False,
            checkpoint: bool = True,
            channels: int | None = None,
            root: str = '',
            config_dir: str = '',
            shape: list[int] | None = None,
            defaults: dict[str, Any] | None = None,
            config: dict[str, Any] | Config | None = None,
            **kwargs: Any) -> None:
        """
        Parameters
        ----------
        net_check : bool
            If layer index should be relative to checkpoint layers
        name : str
            Name of the subnetwork
        shapes : Shapes
            Shape of the outputs from each layer
        cache : bool, Optional
            If checkpoints should be saved or cleared after forward pass, default = False
        checkpoint : bool, Optional
            If layer index should be relative to checkpoint layers, default = True
        channels : int, Optional
            Number of output channels, won't be used if shape is provided, if channels and
            shape aren't provided, the input dimensions will be preserved
        root : str, Optional
            Root directory to prepend to config file path
        config_dir : str, Optional
            Path to the directory with the network configuration file, won't be used if config is
            provided
        shape : list[int] | None, Optional
            Output shape of the block, will be used if provided; otherwise, channels will be used
        defaults : dict[str, Any] | None, Optional
            Default values for the parameters for each type of layer
        config : dict[str, Any] | Config | None, Optional
            Network configuration dictionary
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._cache: bool = cache
        self._checkpoint: bool = checkpoint or net_check
        self.net: Network
        defaults = defaults or {}
        shapes.append(shapes[-1].copy())

        # Subnetwork output if provided
        if shape:
            shapes[-1] = shape
        elif channels:
            shapes[-1][0] = channels

        # Create subnetwork
        self.net = Network(
            name,
            Config.from_dict(config) if isinstance(config, dict) else (config or config_dir),
            shapes[-2],
            shapes[-1],
            suppress_warning=True,
            root=root,
            defaults=defaults | {'checkpoints': self._checkpoint},
        )
        shapes[-1] = self.net.shapes[-1]

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {
            'checkpoint': self._checkpoint,
            'name': self.net.name,
            'shape': self.net.shapes[-1],
            'config': self.net.get_config(True),
        }

    def clear_checkpoints(self) -> None:
        self.net.clear_checkpoints()

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'name={self.net.name}, checkpoint={self._checkpoint}, shape={self.net.shapes[-1]}'

    def forward(self, x: TensorListLike, *_: Any, **__: Any) -> TensorListLike:
        """
        Forward pass for the Composite layer.

        Parameters
        ----------
        x : TensorListT
            Input tensor(s) with dtype float32 and shape (N,...), where N is batch size

        Returns
        -------
        TensorListT
            Output tensor(s) from the subnetwork with dtype float32 and shape (N,...)
        """
        x = self.net(x)

        if not self._cache:
            self.clear_checkpoints()
        return x

    def to(self, *args: Any, **kwargs: Any) -> Self:
        self.net.to(*args, **kwargs)
        return self


def _create_network(
        config: str | Config,
        in_shape: list[int] | list[list[int]],
        out_shape: list[int],
        *,
        suppress_warning: bool = False,
        root: str = '',
        defaults: dict[str, Any] | None = None,
) -> tuple[TypedModuleList[layers.BaseLayer], Shapes, Shapes, Config]:
    """
    Creates a network from a config file

    Parameters
    ----------
    config : str | dict[str, list[dict[str, Any]] | dict[str, Any]]
        Path to the config file or configuration dictionary
    in_shape : list[int] | list[list[int]]
        Shape of the input tensor(s), excluding batch size
    out_shape : list[int]
        Shape of the output tensor, excluding batch size
    suppress_warning : bool, Optional
        If output shape mismatch warning should be suppressed, default = False
    root : str, Optional
        Root directory to prepend to config file path
    defaults : dict[str, Any] | None, Optional
        Default values for the parameters for each type of layer

    Returns
    -------
    module_list : TypedModuleList[BaseLayer]
        Layers in the network with parameters
    shapes : Shapes
        Layer output shapes
    check_shapes : Shapes
        Checkpoint output shapes
    config : Config
        Network construction
    """
    layer: dict[str, Any]
    file: TextIO
    logger: log.Logger = log.getLogger(__name__)
    module_list: TypedModuleList[layers.BaseLayer] = TypedModuleList()
    shapes: Shapes = Shapes([in_shape])
    check_shapes: Shapes = Shapes([])
    layer_class: type[layers.BaseLayer]
    defaults = defaults or {}

    # Load network configuration file
    if isinstance(config, str):
        config += '' if '.json' in config else '.json'

        with (utils.suppress_logger_warnings(configs.__name__)
              if suppress_warning else nullcontext(),
              open(os.path.join(root, config), 'r', encoding='utf-8') as file):
            config = Config.from_dict(json.load(file), new_fields=True)

    if isinstance(config, dict):
        config = Config.from_dict(config, new_fields=True)
        warn(
            'Passing configuration as a dictionary is deprecated, please pass a Config '
            'object or path to the configuration file instead',
            DeprecationWarning,
            stacklevel=2,
        )

    assert isinstance(config, Config)
    config.net.layers = utils.deep_merge(config.net.layers, defaults)

    # Create layers
    for i, layer in enumerate(config.layers.copy()):
        if layer['type'] in config.net.layers:
            layer = config.net.layers[layer['type']] | layer

        if layer['type'] == 'Composite':
            layer_class = Composite
        elif layer['type'] == 'Branch':
            layer_class = Branch

            if layer.get('inherit_defaults', True):
                layer['defaults'] = config.net.layers | layer.get('defaults', {})
        else:
            layer_class = getattr(layers, layer['type'])

        if layer['type'] == 'AdaptivePool' and 'shapes' in layer:
            layer['shape'] = layer.pop('shapes')

        try:
            module_list.append(layer_class(
                net_check=config.net.checkpoints,
                idx=i,
                root=root,
                net_out=out_shape,
                shapes=shapes,
                check_shapes=check_shapes,
                **layer,
            ))
        except Exception as e:
            raise type(e)(f'{e}\nError in {layer["type"]} (layer {i})') from e

        if layer_class == Composite:
            config.layers[i]['config'] = cast(Composite, module_list[-1]).net.get_config(True)

    # Check network output dimensions
    if not suppress_warning and shapes.get(-1, True) != out_shape:
        logger.warning(
            f'Network output shape {shapes.get(-1, True)} != data output shape {out_shape}'
        )
    return module_list, shapes, check_shapes, config


__all__ = ['BaseNetwork', 'CompatibleNetwork', 'Network', 'Branch', 'Composite']
