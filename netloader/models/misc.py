"""
Custom network models.
"""
import logging as log
from itertools import repeat
from typing import Any, Literal, Self, overload

from torch import nn

from netloader.utils.types import TensorListLike
from netloader.utils import Config, TypedModuleList
from netloader.network import BaseNetwork, CompatibleNetwork, Network


class MultiNetwork(BaseNetwork[TypedModuleList[CompatibleNetwork | Network]]):
    """
    Network class to hold multiple networks and train them together.

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
    layers : TypedModuleList[CompatibleNetwork | Network]
        Network construction
    """
    _cache: bool

    @overload
    def __init__(
            self,
            name: str,
            *nets: nn.Module | nn.ModuleList | CompatibleNetwork | Network,
            save_outputs: bool = ...) -> None: ...

    @overload
    def __init__(
            self,
            name: str,
            *nets: str | nn.Module | nn.ModuleList | CompatibleNetwork | Network | Config,
            save_outputs: bool = ...,
            suppress_warning: bool = ...,
            root: str = ...,
            config: str = ...,
            in_shape: list[int] | list[list[int]] | tuple[int, ...] | None = ...,
            out_shapes: list[list[int]] | None = ...,
            defaults: list[dict[str, Any]] | dict[str, Any] | None = ...) -> None: ...

    def __init__(
            self,
            name: str,
            *nets: str | nn.Module | nn.ModuleList | CompatibleNetwork | Network | Config,
            save_outputs: bool = False,
            suppress_warning: bool = False,
            root: str = '',
            config: str = '',
            in_shape: list[int] | list[list[int]] | tuple[int, ...] | None = None,
            out_shapes: list[list[int]] | None = None,
            defaults: list[dict[str, Any]] | dict[str, Any] | None = None) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the network, used for saving
        *nets: str | nn.Module | nn.ModuleList | CompatibleNetwork | Network | Config
            Networks to combine sequentially
        save_outputs : bool, Optional
            If outputs from each network should be saved, default = False
        suppress_warning : bool, Optional
            If output shape mismatch warning should be suppressed, default = False
        root : str, Optional
            Root directory to prepend to config file path if any network is a string or Config
        config : str, Optional
            Path to the network config directory if any network is a string
        in_shape : list[int] | list[list[int]] | tuple[int, ...], Optional
            Input shape for the first network in the MultiNetwork if any network is a string or
            Config
        out_shapes : list[list[int]], Optional
            Output shapes for each network in the MultiNetwork if any network is a string or Config
        defaults : list[dict[str, Any]] | dict[str, Any] | None, Optional
            Default values for the parameters for each type of layer for all or each network
        """
        super().__init__()
        self._cache = save_outputs
        self.name = name
        self.layers = TypedModuleList()
        calc_in_shape: bool = True
        net: str | nn.Module | nn.ModuleList | CompatibleNetwork | Network | Config
        builds: list[bool] = [isinstance(net, (str, Config)) for net in nets]
        out_shape: list[int] | None
        default: dict[str, Any] | None
        defaults = repeat(None) if defaults is None else defaults if isinstance(defaults, list) \
            else [defaults] * len(nets)

        if out_shapes and len(out_shapes) != len(nets):
            raise ValueError(f'Length of out_shape list ({len(out_shapes)}) must match number of '
                             f'networks ({len(nets)})')
        if not isinstance(defaults, repeat) and len(defaults) != len(nets):
            raise ValueError(f'Length of defaults list ({len(defaults)}) must match number of '
                             f'networks ({len(nets)})')
        if any(builds) and (in_shape is None or out_shapes is None):
            raise ValueError(f'If any network is a string or Config object, in_shape ({in_shape}) '
                             f'and out_shapes ({out_shapes}) must be provided')
        if not out_shapes and not all(builds) and any(builds):
            calc_in_shape = False
            log.getLogger(__name__).warning(
                'If out_shape is not provided and any network is not a string or Config object, '
                'each string and Config object will be assumed to have the same in_shape')

        for net, out_shape, default in zip(nets, out_shapes or repeat(None), defaults):
            match net:
                case str():
                    assert in_shape is not None
                    self.layers.append(Network(
                        net,
                        config,
                        in_shape,
                        out_shape or [],
                        suppress_warning=suppress_warning,
                        root=root,
                        defaults=default,
                    ))
                case Config():
                    assert in_shape is not None
                    self.layers.append(Network(
                        '',
                        net,
                        in_shape,
                        out_shape or [],
                        suppress_warning=suppress_warning,
                        root=root,
                        defaults=default,
                    ))
                case Network() | CompatibleNetwork():
                    self.layers.append(net)
                case nn.Module():
                    self.layers.append(CompatibleNetwork(net))
                case _:
                    raise TypeError(f'Unsupported network type ({type(net)})')

            if calc_in_shape and isinstance(self.layers[-1], Network):
                in_shape = self.layers[-1].shapes.get(-1, True)
            elif calc_in_shape:
                in_shape = out_shape

    @overload
    def __getitem__(self, item: int) -> CompatibleNetwork | Network: ...

    @overload
    def __getitem__(self, item: slice) -> TypedModuleList[CompatibleNetwork | Network]: ...

    def __getitem__(
            self,
            item: int | slice,
    ) -> CompatibleNetwork | Network | TypedModuleList[CompatibleNetwork | Network]:
        """
        Returns the network(s) at the given index or slice.

        Parameters
        ----------
        item : int | slice
            Index or slice of the network(s) to return

        Returns
        -------
        CompatibleNetwork | Network | TypedModuleList[CompatibleNetwork | Network]
            The network(s) at the given index or slice
        """
        return self.layers[item]

    def __getstate__(self) -> dict[str, Any]:
        return super().__getstate__() | {'save_outputs': self._cache, 'layers': list(self.layers)}

    def __len__(self) -> int:
        return len(self.layers)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._cache = state.get('save_outputs', False)
        self.layers = TypedModuleList(state.get('layers', state.get('net')))

    def forward(self, x: TensorListLike) -> TensorListLike:
        """
        Forward pass through all networks in the MultiNetwork.

        Parameters
        ----------
        x : TensorListLike
            Input tensor(s) with shape (N,...) and type float, where N is the
            batch size

        Returns
        -------
        TensorListLike
            Output tensor from the network with shape (N,...) and type float
        """
        net: CompatibleNetwork | Network
        self.checkpoints = []

        for i, net in enumerate(self.layers):
            x = net(x)
            self.checkpoints.extend(net.checkpoints)

            if self._cache and i != len(self.layers) - 1:
                self.checkpoints.append(x.clone())
        return x

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
        net: CompatibleNetwork | Network
        config = Config(layers=[net.get_config() for net in self.layers])
        return config.to_dict() if dict_ else config

    def to(self, *args: Any, **kwargs: Any) -> Self:
        super().to(*args, **kwargs)
        self.layers.to(*args, **kwargs)
        return self
