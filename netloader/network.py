"""
Constructs a network from layers and can load weights to resume network training
"""
import os
import json
import logging as log
from warnings import warn
from contextlib import nullcontext
from typing import Any, TextIO, Self, Iterator, cast, overload

import torch
from torch import nn, Tensor
from zuko.distributions import NormalizingFlow

import netloader
from netloader import layers
from netloader.utils.configs import Config
from netloader.utils.types import TensorListLike
from netloader.utils import configs, Shapes, deep_merge, suppress_logger_warnings


class CompatibleNetwork(nn.Module):
    """
    A wrapper for nn.Module that ensures compatibility with BaseNetwork by adding required
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
    net : nn.ModuleList
        Network construction

    Methods
    -------
    forward(x) -> TensorListLike
        Forward pass of the network
    """
    def __init__(self, net: nn.Module | nn.ModuleList, *, name: str = '') -> None:
        """
        Parameters
        ----------
        net : nn.Module | nn.ModuleList
            The neural network module to wrap or list of layers in the network
        name : str, default = ''
            Name of the network, used for saving
        """
        super().__init__()
        self.name: str
        self.version: str = netloader.__version__
        self.checkpoints: list[TensorListLike] = []
        self.kl_loss: Tensor = torch.tensor(0.)
        self.net: nn.ModuleList = nn.ModuleList(net.children()) \
            if isinstance(net, nn.Module) else net

        if name:
            self.name = name
        elif hasattr(net, 'name') and isinstance(net.name, str):
            self.name = net.name
        elif isinstance(net, nn.ModuleList):
            self.name = self.__class__.__name__
        else:
            self.name = net.__class__.__name__

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the network for pickling

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the network
        """
        return {
            'name': self.name,
            'version': netloader.__version__,
            'net': self.state_dict(),
        }

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
        self.version = state['version'] if 'version' in state else '<3.9.0'
        self.checkpoints = []
        self.kl_loss = torch.tensor(0.)
        self.load_state_dict(state['net'])

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

        for layer in self.net:
            x = layer(x)
        return x


class Network(nn.Module):
    """
    Constructs a neural network from a configuration file.

    Attributes
    ----------
    layer_num : int
        Number of layers to use, if None use all layers
    group : int
        Which group is the active group if a layer has the group attribute
    name : str
        Name of the network, used for saving
    version : str
        NetLoader version string
    checkpoints : list[TensorListLike]
        Outputs from each checkpoint with each Tensor having shape (N,...) and type float, where N
        is the batch size
    kl_loss : Tensor
        KL divergence loss on the latent space of shape (1) and type float, if using a sample layer
    net : nn.ModuleList
        Network construction
    check_shapes : Shapes
        Checkpoint output shapes
    shapes : Shapes
        Layer output shapes
    config : Config
        Network configuration

    Methods
    -------
    forward(x) -> NormalizingFlow | TensorListLike
        Forward pass of the network
    to(*args, **kwargs) -> Network
        Moves and/or casts the parameters and buffers
    """
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
            Path to the network _config directory or configuration dictionary
        in_shape : list[int] | list[list[int]] | tuple[int, ...]
            Shape of the input tensor(s), excluding batch size
        out_shape : list[int]
            Shape of the output tensor, excluding batch size
        suppress_warning : bool, default = False
            If output shape mismatch warning should be suppressed
        root : str, default = ''
            Root directory to prepend to _config file path
        defaults : dict[str, Any], default = None
            Default values for the parameters for each type of layer
        """
        super().__init__()
        self._checkpoints: bool
        self.group: int = 0
        self.layer_num: int | None = None
        self.name: str = name
        self.version: str = netloader.__version__
        self.checkpoints: list[TensorListLike] = []
        self.kl_loss: Tensor = torch.tensor(0.)
        self.net: nn.ModuleList
        self.shapes: Shapes
        self.check_shapes: Shapes
        self._config: Config
        layer: layers.BaseLayer

        if '.json' not in self.name:
            self.name += '.json'

        # Construct layers in network
        self.net, self.shapes, self.check_shapes, self._config = _create_network(
            os.path.join(config, self.name) if isinstance(config, str) else config,
            in_shape if isinstance(in_shape, list) else list(in_shape),
            list(out_shape),
            suppress_warning=suppress_warning,
            root=root,
            defaults=defaults,
        )
        self._checkpoints = self._config.net.checkpoints
        self.name = self.name.replace('.json', '')
        self._config.layers = [layer.__getstate__() for layer in self.net]

        # Adds Network class to list of safe PyTorch classes when loading saved networks
        torch.serialization.add_safe_globals([self.__class__])

    def __len__(self) -> int:
        """
        Returns the number of layers in the network.

        Returns
        -------
        int
            Number of layers in the network
        """
        return len(self.net[:self.layer_num])

    @overload
    def __getitem__(self, idx: int) -> nn.Module: ...

    @overload
    def __getitem__(self, idx: slice) -> nn.ModuleList: ...

    def __getitem__(self, idx: int | slice) -> nn.Module | nn.ModuleList:
        """
        Returns the layer at the specified index.

        Parameters
        ----------
        idx : int | slice
            Index or slice of the layer(s) to return

        Returns
        -------
        nn.Module | nn.ModuleList
            Layer(s) at the specified index or slice
        """
        return self.net[idx]

    def __iter__(self) -> Iterator[nn.Module]:
        """
        Returns an iterator over the layers in the network.

        Returns
        -------
        Iterator[nn.Module]
            Iterator over the layers in the network
        """
        i: int

        for i in range(len(self)):
            yield self[i]

    def __getstate__(self) -> dict[str, Any]:
        """
        Returns a dictionary containing the state of the network for pickling.

        Returns
        -------
        dict[str, Any]
            Dictionary containing the state of the network
        """
        return {
            'group': self.group,
            'layer_num': self.layer_num,
            'name': self.name,
            'version': netloader.__version__,
            'shapes': list(self.shapes),
            'config': self.get_config(True),
            'net': self.state_dict(),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Sets the state of the network for pickling.

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the state of the network
        """
        super().__init__()
        shapes = Shapes(state['shapes'])
        self.group = state['group']
        self.layer_num = state['layer_num']
        self.name = state['name']
        self.version = state.get('version', '<3.9.0')
        self.checkpoints = []
        self.kl_loss = torch.tensor(0.)

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
            self.net = state['_modules']
            return

        self.net, self.shapes, self.check_shapes, self._config = _create_network(
            Config.from_dict(state['config']),
            shapes.get(0, True),
            shapes[-1],
            suppress_warning=True,
        )
        self._checkpoints = self._config.net.checkpoints

        try:
            self.load_state_dict(state['net'])
        except RuntimeError:
            if 'net' not in list(state['net'].keys())[0]:
                warn(
                    'Network state is saved in old non-weights safe format and is '
                    'deprecated, please resave the network in the new format using net.save()',
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                log.getLogger().warning('Failed to load network, incorrect layer names, will try '
                                        'to match weights.')

            self.load_state_dict(dict(zip(self.state_dict(), state['net'].values())))

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

    def get_config(self, dict_: bool = True) -> dict[str, Any] | Config:
        """
        Returns the network configuration.

        Returns
        -------
        dict[str, Any] | Config
            Network configuration
        """
        return self._config.to_dict() if dict_ else self._config

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
        distributions.NormalizingFlow | TensorListLike
            Output tensor from the network with shape (N,...) and type float, or NormalizingFlow if
            the last layer is from layers.flows
        """
        i: int
        outputs: list[TensorListLike] = []
        e: Exception
        layer: layers.BaseLayer
        self.checkpoints = []

        if not self._checkpoints or any(isinstance(layer, layers.Unpack) for layer in self.net):
            outputs = [x]

        for i, layer in enumerate(self.net[:self.layer_num]):
            if layer.group not in (0, self.group):
                outputs.append(torch.tensor([]))

                if isinstance(layer, layers.Checkpoint):
                    self.checkpoints.append(torch.tensor([]))

            try:
                x = layer(x, outputs=outputs, checkpoints=self.checkpoints, net=self)
            except Exception as e:
                raise type(e)(f'{e}\nError in {layer.__class__.__name__} (layer {i})') from e

            if not self._checkpoints:
                outputs.append(x)
        return x

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        i: int
        checkpoint: TensorListLike
        layer: layers.BaseLayer
        self.kl_loss = self.kl_loss.to(*args, **kwargs)

        for layer in cast(list[layers.BaseLayer], list(self.net)):
            layer.to(*args, **kwargs)

        for i, checkpoint in enumerate(self.checkpoints):
            if hasattr(checkpoint, 'to'):
                self.checkpoints[i] = checkpoint.to(*args, **kwargs)
            else:
                log.getLogger().warning(f'Failed to move checkpoint '
                                        f'{checkpoint.__class__.__name__} to device')
        return self


class Composite(layers.BaseLayer):
    """
    Creates a subnetwork from a configuration file.

    Attributes
    ----------
    layers : list[Module] | Sequential
        Layers part of the parent layer to loop through in the forward pass
    net : Network
        Subnetwork for the Composite layer

    Methods
    -------
    forward(x) -> TensorListT
        Forward pass of the subnetwork
    to(*args, **kwargs) -> Self
        Moves and/or casts the parameters and buffers
    extra_repr() -> str
        Displays layer parameters when printing the network
    """
    def __init__(
            self,
            net_check: bool,
            name: str,
            shapes: Shapes,
            *,
            checkpoint: bool = True,
            channels: int | None = None,
            root: str = '',
            config_dir: str = '',
            shape: list[int] | None = None,
            defaults: dict[str, Any] | None = None,
            config: Config | None = None,
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
        checkpoint : bool, default = True
            If layer index should be relative to checkpoint layers
        channels : int, optional
            Number of output channels, won't be used if shape is provided, if channels and
            shape aren't provided, the input dimensions will be preserved
        root : str, default = ''
            Root directory to prepend to _config file path
        config_dir : str, default = ''
            Path to the directory with the network configuration file, won't be used if _config is
            provided
        shape : list[int], optional
            Output shape of the block, will be used if provided; otherwise, channels will be used
        defaults : dict[str, Any], default = None
            Default values for the parameters for each type of layer
        config : Config | None, default = None
            Network configuration dictionary
        **kwargs
            Leftover parameters to pass to base layer for checking
        """
        super().__init__(**kwargs)
        self._checkpoint: bool
        self.net: Network
        self._checkpoint = checkpoint or net_check
        shapes.append(shapes[-1].copy())

        if defaults is None:
            defaults = {}

        # Subnetwork output if provided
        if shape:
            shapes[-1] = shape
        elif channels:
            shapes[-1][0] = channels

        # Create subnetwork
        self.net = Network(
            name,
            config or config_dir,
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

    def forward(
            self,
            x: TensorListLike,
            *_: Any,
            **__: Any) -> TensorListLike:
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
        return self.net(x)

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """
        Moves and/or casts the parameters and buffers.

        Parameters
        ----------
        *args
            Arguments to pass to torch.Tensor.to
        **kwargs
            Keyword arguments to pass to torch.Tensor.to
        """
        self.net.to(*args, **kwargs)
        return self

    def extra_repr(self) -> str:
        """
        Displays layer parameters when printing the network.

        Returns
        -------
        str
            Layer parameters
        """
        return f'name={self.net.name}, checkpoint={self._checkpoint}, shape={self.net.shapes[-1]}'


def _create_network(
        config: str | Config,
        in_shape: list[int] | list[list[int]],
        out_shape: list[int],
        *,
        suppress_warning: bool = False,
        root: str = '',
        defaults: dict[str, Any] | None = None) -> tuple[nn.ModuleList, Shapes, Shapes, Config]:
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
    suppress_warning : bool, default = False
        If output shape mismatch warning should be suppressed
    root : str, default = ''
        Root directory to prepend to config file path
    defaults : dict[str, Any], default = None
        Default values for the parameters for each type of layer

    Returns
    -------
    module_list : nn.ModuleList
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
    module_list: nn.ModuleList = nn.ModuleList()
    shapes: Shapes = Shapes([in_shape])
    check_shapes: Shapes = Shapes([])
    defaults = defaults or {}

    # Load network configuration file
    if isinstance(config, str):
        config += '' if '.json' in config else '.json'

        with (suppress_logger_warnings(configs.__name__) if suppress_warning else nullcontext(),
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
    config.net.layers = deep_merge(config.net.layers, defaults)

    # Create layers
    for i, layer in enumerate(config.layers.copy()):
        if layer['type'] in config.net.layers:
            layer = config.net.layers[layer['type']] | layer

        if layer['type'] == 'Composite':
            layer_class = Composite
        else:
            layer_class = getattr(layers, layer['type'])

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
        except ValueError:
            logger.error(f"Error in {layer['type']} (layer {i})")
            raise

        if layer_class == Composite:
            config.layers[i]['config'] = cast(Composite, module_list[-1]).net.get_config(True)

    # Check network output dimensions
    if not suppress_warning and shapes[-1] != out_shape:
        logger.warning(
            f'Network output shape {shapes[-1]} != data output shape {out_shape}'
        )
    return module_list, shapes, check_shapes, config
