"""
Configuration dataclasses for the network and layers.
"""
import logging as log
from dataclasses import dataclass, field
from typing import TypeVar, Any, Literal, get_type_hints

from netloader import layers


ConfigT = TypeVar('ConfigT', bound='BaseConfig')


@dataclass
class BaseConfig:
    """
    Base dataclass to hold configuration parameters.

    Methods
    -------
    from_dict(config: dict[str, Any]) -> ConfigT
        Creates a BaseConfig object from a dictionary
    get_changed_fields() -> set[str]
        Gets the set of changed fields in the dataclass
    merge(other: ConfigT) -> None
        Merges another dataclass into this one
    merge_dict(other: dict[str, Any]) -> None
        Merges a dictionary into this dataclass
    to_dict() -> dict[str, Any]
        Converts the dataclass to a dictionary
    """
    _changed_fields: set[str] = field(default_factory=set, init=False, repr=False)

    def __getattr__(self, item: str) -> Any:
        """
        Gets an attribute of the dataclass.

        Parameters
        ----------
        item : str
            Key of the attribute

        Returns
        -------
        Any
            Value of the attribute
        """
        item = item.replace('-', '_')
        return super().__getattribute__(item)

    def __setattr__(self, key: str, value: Any) -> None:
        """
        Sets an attribute of the dataclass.

        Parameters
        ----------
        key : str
            Key of the attribute
        value : Any
            Value to set
        """
        if key.startswith('_'):
            return super().__setattr__(key, value)

        key = key.replace('-', '_')
        hint: type = get_type_hints(self)[key]

        if getattr(hint, '__origin__', None) is Literal:
            self._check_literal(key, value, getattr(hint, '__args__', ()))

        if value != self.__dataclass_fields__[key].default:
            self._changed_fields.add(key)
        return super().__setattr__(key, value)

    @staticmethod
    def _check_literal(key: str, value: Any, options: tuple[Any, ...]) -> None:
        """
        Checks if a value is in a list of options.

        Parameters
        ----------
        key : str
            Key of the value
        value : Any
            Value to check
        options : list[Any]
            List of options
        """
        if value not in options:
            raise ValueError(f'Invalid value for {key} ({value}), must be one of {options}')

    @classmethod
    def from_dict(cls: type[ConfigT], config: dict[str, Any], new_fields: bool = False) -> ConfigT:
        """
        Creates a BaseConfig object from a dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary
        new_fields : bool, default = False
            Whether to allow new fields in the dictionary

        Returns
        -------
        ConfigT
            Configuration dataclass
        """
        instance: ConfigT = type.__call__(cls)
        instance.merge_dict(config, new_fields=new_fields)
        return instance

    def get_changed_fields(self) -> set[str]:
        """
        Gets the set of changed fields in the dataclass.

        Returns
        -------
        set[str]
            Set of changed fields
        """
        return self._changed_fields

    def merge(self: ConfigT, other: ConfigT, new_fields: bool = False) -> None:
        """
        Merges another dataclass into this one.

        Parameters
        ----------
        other : ConfigT
            Configuration dataclass to merge
        new_fields : bool, default = False
            Whether to allow new fields in the other dataclass
        """
        key: str

        for key in other.get_changed_fields():
            if not hasattr(self, key) and not new_fields:
                raise AttributeError(f'Unknown config key ({key}) and new_fields is False')

            if hasattr(self, key) and isinstance(getattr(self, key), BaseConfig):
                getattr(self, key).merge(getattr(other, key), new_fields=new_fields)
            else:
                setattr(self, key, getattr(other, key))

    def merge_dict(self, other: dict[str, Any], new_fields: bool = False) -> None:
        """
        Merges a dictionary into this dataclass.

        Parameters
        ----------
        other : dict[str, Any]
            Configuration dictionary to merge
        new_fields : bool, default = False
            Whether to allow new fields in the dictionary
        """
        key: str
        value: Any

        for key, value in other.items():
            key.replace('-', '_')

            if not hasattr(self, key) and not new_fields:
                raise AttributeError(f'Unknown config key ({key}) and new_fields is False')

            if hasattr(self, key) and isinstance(getattr(self, key), BaseConfig):
                getattr(self, key).merge_dict(value, new_fields=new_fields)
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the dataclass to a dictionary.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary
        """
        key: str
        result: dict[str, Any] = {}
        value: Any

        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, BaseConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


@dataclass
class NetConfig(BaseConfig):
    """
    Dataclass to hold global network configuration parameters.

    Attributes
    ----------
    checkpoints : bool
        Whether to use checkpoint layers in the network
    paper : str
        Link to the paper describing the network
    github : str
        Link to the GitHub repository for the network
    description : str
        Description of the network
    layers : dict[str, Any]
        Dictionary of default layer configurations
    """
    checkpoints: bool = False
    paper: str = ''
    github: str = ''
    description: str = ''
    layers: dict[str, Any] = field(default_factory=dict)

    def merge_dict(self, other: dict[str, Any], new_fields: bool = False) -> None:
        key: str
        value: Any

        if 'layers' in other:
            self.layers.update(other.pop('layers'))

        for key, value in other.items():
            if hasattr(layers, key) or key == 'Composite':
                self.layers[key] = value
            elif not hasattr(self, key) and not new_fields:
                raise AttributeError(f'Unknown net config key ({key}) and new_fields is False')
            elif hasattr(NetConfig, key) and isinstance(getattr(self, key), BaseConfig):
                getattr(self, key).merge_dict(value, new_fields=new_fields)
            elif hasattr(NetConfig, key):
                setattr(self, key, value)
            else:
                log.getLogger(__name__).warning(f'Unknown net config key ({key})')


@dataclass
class Config(BaseConfig):
    """
    Dataclass to hold global network and layer configuration parameters.

    Attributes
    ----------
    layers : list[dict[str, Any]]
        List of layers
    net : NetConfig
        Network configuration
    """
    layers: list[dict[str, Any]] = field(default_factory=list)
    net: NetConfig = field(default_factory=NetConfig)
