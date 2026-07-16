"""
Utility functions for the documentation.
"""
from __future__ import annotations

import re
import typing
import inspect
from types import ModuleType
from importlib import import_module
from typing import Any, Callable, Sequence, TypeVar, ForwardRef, Type, get_args, get_origin, cast

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.ext.autodoc import DataDocumenter
from sphinx.addnodes import pending_xref

# NumPyDoc cross-reference aliases
NUMPYDOC_ALIASES: dict[str, str] = {
    # Builtins / typing
    'Any': ':class:`~typing.Any`',
    'Self': ':class:`~typing.Self`',
    'Literal': ':class:`~typing.Literal`',
    'Generic': ':class:`~typing.Generic`',
    'Sequence': ':class:`~typing.Sequence`',
    'Callable': ':class:`~typing.Callable`',
    'Protocol': ':class:`~typing.Protocol`',
    'Iterator': ':class:`~typing.Iterator`',
    'ModuleType': ':class:`~types.ModuleType`',
    'None': ':obj:`~types.NoneType`',
    'NoneType': ':obj:`~types.NoneType`',
    'Ellipsis': ':obj:`~types.EllipsisType`',

    # NumPy
    'dtype': ':class:`~numpy.dtype`',
    'ndarray': ':class:`~numpy.ndarray`',

    # PyTorch
    'Tensor': ':class:`~torch.Tensor`',
    'Module': ':class:`~torch.nn.Module`',
    'Subset': ':class:`~torch.utils.data.Subset`',
    'Sequential': ':class:`~torch.nn.Sequential`',
    'ModuleList': ':class:`~torch.nn.ModuleList`',
    'Optimizer': ':class:`~torch.optim.Optimizer`',
    'Dataset': ':class:`~torch.utils.data.Dataset`',
    'DataLoader': ':class:`~torch.utils.data.DataLoader`',
    'Normal': ':class:`~torch.distributions.normal.Normal`',
    'LRScheduler': ':class:`~torch.optim.lr_scheduler.LRScheduler`',
    'ParamsT': ':type:`~torch.optim.optimizer.ParamsT`',

    # Zuko
    'NormalizingFlow': ':class:`~zuko.distributions.NormalizingFlow`',

    # Netloader
    'Data': ':class:`~netloader.data.Data`',
    'DataList': ':class:`~netloader.data.DataList`',
    'Network': ':class:`~netloader.network.Network`',
    'Shapes': ':class:`~netloader.utils.utils.Shapes`',
    'Config': ':class:`~netloader.utils.configs.Config`',
    'BaseDataset': ':class:`~netloader.data.BaseDataset`',
    'BaseLoss': ':class:`~netloader.loss_funcs.BaseLoss`',
    'Pack': ':class:`~netloader.layers.multi_layer.Pack`',
    'BaseNetwork': ':class:`~netloader.network.BaseNetwork`',
    'BaseLayer': ':class:`~netloader.layers.base.BaseLayer`',
    'NetConfig': ':class:`~netloader.utils.configs.NetConfig`',
    'BaseConfig': ':class:`~netloader.utils.configs.BaseConfig`',
    'Shortcut': ':class:`~netloader.layers.multi_layer.Shortcut`',
    'BaseTransform': ':class:`~netloader.transforms.BaseTransform`',
    'Concatenate': ':class:`~netloader.layers.multi_layer.Concatenate`',
    'TypedModuleList': ':class:`~netloader.utils.utils.TypedModuleList`',
    'CompatibleNetwork': ':class:`~netloader.network.CompatibleNetwork`',
    'DatasetProtocol': ':class:`~netloader.utils.types.DatasetProtocol`',
    'Param': ':type:`~netloader.utils.types.Param`',
    'DataLike': ':type:`~netloader.utils.types.DataLike`',
    'ArrayLike': ':type:`~netloader.utils.types.ArrayLike`',
    'TensorLike': ':type:`~netloader.utils.types.TensorLike`',
    'NDArrayLike': ':type:`~netloader.utils.types.NDArrayLike`',
    'DataListLike': ':type:`~netloader.utils.types.DataListLike`',
    'TensorListLike': ':type:`~netloader.utils.types.TensorListLike`',
    'NDArrayListLike': ':type:`~netloader.utils.types.NDArrayListLike`',
    'T': ':type:`~netloader.utils.types.T`',
    'P': ':type:`~netloader.utils.types.P`',
    'DataT': ':type:`~netloader.utils.types.DataT`',
    'ArrayT': ':type:`~netloader.utils.types.ArrayT`',
    'ModuleT': ':type:`~netloader.utils.types.ModuleT`',
    'TensorT': ':type:`~netloader.utils.types.TensorT`',
    'NDArrayT': ':type:`~netloader.utils.types.NDArrayT`',
    'DatasetT': ':type:`~netloader.utils.types.DatasetT`',
    'DataListT': ':type:`~netloader.utils.types.DataListT`',
    'TensorListT': ':type:`~netloader.utils.types.TensorListT`',
    'ModuleListT': ':type:`~netloader.utils.types.ModuleListT`',
    'LossCT': ':type:`~netloader.utils.types.LossCT`',
    'ArrayCT': ':type:`~netloader.utils.types.ArrayCT`',
    'InArrayCT': ':type:`~netloader.transforms.InArrayCT`',
    'OutArrayCT': ':type:`~netloader.transforms.OutArrayCT`',
    'TensorLossCT': ':type:`~netloader.utils.types.TensorLossCT`',
}
OBJ_REF: dict[str, type]


class TypeDocumenter(DataDocumenter):
    """
    Custom documenter for type aliases, TypeVars, & param specs.
    """
    priority = DataDocumenter.priority + 1
    objtype = 'cleantype'
    directivetype = 'data'

    @staticmethod
    def _types(
            lines: list[str],
            obj: type,
            aliases: dict[str, str] | None = None,
            obj_ref: dict[str, type] | None = None) -> None:
        """
        Formats TypeVars, TypeAliases, and other type definitions.

        Parameters
        ----------
        lines : list[str]
            List of lines to append the formatted type definition to
        obj : type
            The type object to format
        aliases : dict[str, str] | None, Optional
            A dictionary of aliases for cross-referencing types
        obj_ref : dict[str, type] | None, Optional
            A dictionary mapping type names to their corresponding type objects for
            cross-referencing
        """
        args: str
        constraints: tuple[type, ...]
        bound: type
        constraint: type
        aliases = aliases or {}
        lines.clear()

        if isinstance(obj, TypeVar):
            bound = getattr(obj, '__bound__', None)
            constraints = getattr(obj, '__constraints__', ())

            if constraints:
                args = ', '.join(format_type(
                    constraint,
                    aliases=aliases,
                    obj_ref=obj_ref,
                ) for constraint in constraints)
                lines.append(rf':class:`~typing.TypeVar`\[{args}\]')
            elif bound:
                args = format_type(bound, aliases=aliases, obj_ref=obj_ref)
                lines.append(rf':class:`~typing.TypeVar`\[bound = {args}\]')
            else:
                lines.append(':class:`~typing.TypeVar`')
            return
        if isinstance(obj, typing.ParamSpec):
            lines.append(':class:`~typing.ParamSpec`' + (
                '' if obj.__default__ is typing.NoDefault else rf'\[default = {obj.__default__}\]'
            ))
            return

        lines.append(f'Alias of {format_type(obj, aliases=aliases, obj_ref=obj_ref)}')

    def should_suppress_value_header(self) -> bool:
        return True

    def add_content(self, more_content: StringList | None) -> None:
        super().update_content(more_content)
        lines: list[str] = []
        self._types(lines, self.object, aliases=NUMPYDOC_ALIASES, obj_ref=OBJ_REF)
        self.add_line(lines[0], '')


class BaseAutoSignature(Directive):
    """
    Directive to generate class initialisation signature based on a base class for either a specific
    class or module of classes.

    Attributes
    ----------
    has_content : bool
        Whether the directive has content
    required_arguments : int
        Number of required arguments for the directive
    name : str
        Name of the directive
    base_class : str
        Name of the base class
    exclude : set[str]
        Set of initialisation arguments to exclude from the directive
    class_exclude : set[str]
        Set of classes to exclude from the directive
    """
    has_content = False
    required_arguments = 1
    name: str = 'autosig'
    base_class: str = ''
    exclude: set[str] = {}
    class_exclude: set[str] = {}

    def run(self) -> Sequence[nodes.Node]:
        """
        Run the directive to generate the class(es) initialisation signature.

        Returns
        -------
        Sequence[Node]
            List of nodes generated by the directive
        """
        mod_name: str
        class_name: str
        name: str = self.arguments[0].strip()
        output: list[nodes.Node] = []
        class_: type
        mod: ModuleType | None = None

        try:
            mod = import_module(name)
        except (ModuleNotFoundError, ImportError):
            pass

        if mod:
            for _, class_ in inspect.getmembers(mod, inspect.isclass):
                if class_.__module__ != mod.__name__ or class_.__name__ in self.class_exclude:
                    continue

                output.extend(self.render_class(class_))
            return output

        mod_name, class_name = name.rsplit('.', maxsplit=1)

        try:
            mod = import_module(mod_name)
            class_ = getattr(mod, class_name)

            if not inspect.isclass(class_):
                return [self.state_machine.reporter.error(
                    f'Object {name} is not a class',
                    line=self.lineno,
                )]
        except (ModuleNotFoundError, ImportError, AttributeError):
            return [self.state_machine.reporter.error(
                f'{self.name}: could not load class/module ({name})',
                line=self.lineno
            )]

        output.extend(self.render_class(class_))
        return output

    def render_class(self, class_: type) -> Sequence[nodes.Node]:
        """
        Render the class initialisation signature based on the base class.

        Parameters
        ----------
        class_ : type
            Class to render the initialisation signature for

        Returns
        -------
        Sequence[Node]
            List of nodes generated by the directive
        """
        i: int
        line: str
        param_name: str = ''
        doc: str = inspect.getdoc(class_) or ''
        params_doc: list[str] = (inspect.getdoc(class_.__init__) or '').splitlines()
        output: list[nodes.Node] = []
        params: dict[str, str | list[str]] = {}
        para: nodes.paragraph = nodes.paragraph()

        doc = doc.strip().splitlines()[0] if doc else ''
        para += pending_xref(
            '',
            nodes.literal(text=class_.__name__),
            refdomain='py',
            reftype='class',
            reftarget=f'{class_.__module__}.{class_.__name__}',
            refexplicit=False,
        )
        para += nodes.Text(f' : {doc}' if doc else '')
        output.append(para)

        for i, line in enumerate(params_doc):
            if line.strip().lower() == 'parameters' and i != len(params_doc) - 1:
                i += 2
                break
        else:
            self.render_signature(output, {}, class_)
            return output

        for line in params_doc[i:]:
            if not line.strip():
                continue
            if i < len(params_doc) - 1 and params_doc[i + 1].startswith('--'):
                break

            if not line.startswith('  '):
                param_name = line.split(':', maxsplit=1)[0].strip()
                params[param_name] = []
            elif param_name:
                params[param_name].append(line.strip().rsplit(', default = ', maxsplit=1)[0])

        for param_name in params:
            params[param_name] = ' '.join(params[param_name])

        self.render_signature(output, cast(dict[str, str], params), class_)
        return output

    def render_signature(
            self,
            output: list[nodes.Node],
            params: dict[str, str],
            class_: type) -> None:
        """
        Render the class initialisation signature based on the base class.

        Parameters
        ----------
        output : list[Node]
            List of nodes to append the signature to
        params : dict[str, str]
            Dictionary of parameter names and their types
        class_ : type
            Class to render the initialisation signature for
        """
        name: str
        desc: str
        hints: dict[str, type] = typing.get_type_hints(class_.__init__)
        signature: inspect.Signature = inspect.signature(class_.__init__)
        param: inspect.Parameter
        item: nodes.list_item
        para: nodes.paragraph
        list_: nodes.bullet_list = nodes.bullet_list(bullet='')

        for name, desc in params.items():
            name = name.replace('\\', '')
            param = signature.parameters.get(name.replace('*', ''))

            if name in self.exclude:
                continue

            item = nodes.list_item()
            para = nodes.paragraph()
            self.state.nested_parse(StringList([
                f'``{name}`` : ' +
                format_type(
                    hints[name.replace("*", "")],
                    aliases=NUMPYDOC_ALIASES,
                    obj_ref=OBJ_REF,
                ) +
                (f' = ``{param.default}``' if param and param.default not in {param.empty, ''} else
                 " = ``''``" if param and param.default == '' else '') + f' – {desc}'
            ]), 0, para)
            item += para
            list_ += item

        if self.base_class:
            item = nodes.list_item()
            para = nodes.paragraph()
            self.state.nested_parse(StringList([
                f'``**kwargs`` : Additional keyword arguments passed to '
                f':class:`{self.base_class}`.',
            ]), 0, para)
            item += para
            list_ += item

        output.append(list_)


class AutoLossSignature(BaseAutoSignature):
    """
    Directive to generate class initialisation signature for BaseLoss classes.

    Attributes
    ----------
    has_content : bool
        Whether the directive has content
    required_arguments : int
        Number of required arguments for the directive
    name : str
        Name of the directive
    base_class : str
        Name of the base class
    exclude : set[str]
        Set of initialisation arguments to exclude from the directive
    class_exclude : set[str]
        Set of classes to exclude from the directive
    """
    name: str = 'autoloss'
    base_class: str = '~netloader.loss_funcs.BaseLoss'
    exclude: set[str] = {'self', '*args', '**kwargs'}
    class_exclude: set[str] = {'BaseLoss'}


class AutoLayerSignature(BaseAutoSignature):
    """
    Directive to generate class initialisation signature for BaseLayer classes.

    Attributes
    ----------
    has_content : bool
        Whether the directive has content
    required_arguments : int
        Number of required arguments for the directive
    name : str
        Name of the directive
    base_class : str
        Name of the base class
    exclude : set[str]
        Set of initialisation arguments to exclude from the directive
    class_exclude : set[str]
        Set of classes to exclude from the directive
    """
    name: str = 'autolayer'
    base_class: str = '~netloader.layers.base.BaseLayer'
    exclude: set[str] = {
        'self',
        'type',
        'root',
        'net_check',
        'net_out',
        'shapes',
        'check_shapes',
        '**kwargs',
    }


class AutoArchsSignature(BaseAutoSignature):
    """
    Directive to generate class initialisation signature for BaseArchitecture classes.

    Attributes
    ----------
    has_content : bool
        Whether the directive has content
    required_arguments : int
        Number of required arguments for the directive
    name : str
        Name of the directive
    base_class : str
        Name of the base class
    exclude : set[str]
        Set of initialisation arguments to exclude from the directive
    class_exclude : set[str]
        Set of classes to exclude from the directive
    """
    name: str = 'autoarchs'
    base_class: str = '~netloader.architectures.base.BaseArchitecture'
    exclude: set[str] = {
        'self',
        'save_num',
        'states_dir',
        'net',
        'overwrite',
        'mix_precision',
        'save_freq',
        'learning_rate',
        'description',
        'verbose',
        'transform',
        'in_transform',
        'optimiser_kwargs',
        'scheduler_kwargs',
    }


def _numpydoc_module_convert(key: str, value: str) -> Type:
    """
    Convert a Numpy cross-reference key-value to a Python object.

    Parameters
    ----------
    key : str
        The name of the object
    value : str
        The Numpy cross-reference

    Returns
    -------
    Type
        The Python object
    """
    mod: str
    class_: str
    regex: re.Pattern = re.compile(r'^:(?:class|type|func|meth|mod|obj|data):`~?([^`]+)`$')
    match: re.Match | None = regex.match(value.strip())

    if match:
        mod, class_ = match.group(1).rsplit('.', 1)
        return getattr(import_module(mod), class_)
    raise ValueError(f'Invalid reference ({value}) for {key}')


def format_type(
        obj: type,
        aliases: dict[str, str] | None = None,
        obj_ref: dict[str, type] | None = None) -> str:
    """
    Converts a type object into a string representation, using aliases for cross-referencing if
    provided.

    Parameters
    ----------
    obj : type
        The type object to format
    aliases : dict[str, str] | None, Optional
        A dictionary of aliases for cross-referencing types
    obj_ref : dict[str, type] | None, Optional
        A dictionary mapping type names to their corresponding type objects for cross-referencing

    Returns
    -------
    str
        The formatted string representation of the type
    """
    args: str
    base: str
    sub_obj: type
    aliases = aliases or {}

    # Resolve ForwardRefs, Unions, & Literals
    if isinstance(obj, ForwardRef):
        return format_type(obj.evaluate(locals=obj_ref), aliases=aliases, obj_ref=obj_ref)
    if get_origin(obj) is typing.Union:
        return ' | '.join(format_type(
            sub_obj,
            aliases=aliases,
            obj_ref=obj_ref,
        ) for sub_obj in get_args(obj))
    if get_origin(obj) is typing.Literal:
        return '{' + ', '.join(
            sub_obj if isinstance(sub_obj, str) else
            format_type(sub_obj, aliases=aliases, obj_ref=obj_ref) for sub_obj in get_args(obj)
        ) + '}'

    # Generic / parameterized types
    if hasattr(obj, '__origin__') and hasattr(obj, '__args__'):
        args = ', '.join(format_type(
            sub_obj,
            aliases=aliases,
            obj_ref=obj_ref,
        ) for sub_obj in get_args(obj))
        base = getattr(get_origin(obj), '__name__', repr(get_origin(obj)))
        return rf'{aliases.get(base, f":class:`{base}`")}\[{args}\]'

    # Regular classes
    base = getattr(obj, '__name__', repr(obj))
    return aliases.get(base, f':class:`{base}`')


def overload_docstring(
        _: Any,
        what: str,
        name: str,
        obj: type,
        __: Any,
        lines: list[str]) -> None:
    """
    Adds overload signatures to the docstring of a function or method.

    Parameters
    ----------
    what : str
        The type of the object being documented
    name : str
        The fully qualified name of the object being documented
    obj : type
        The object being documented
    lines : list[str]
        The lines of the docstring to be modified
    """
    i: int
    args: str
    overload_lines: list[str] = ['', ':Overloads:', '']
    func: Callable
    overload: Callable
    overloads: typing.Sequence[Callable]
    sig: inspect.Signature

    if not name.startswith('netloader.') or what not in ('function', 'method', 'class'):
        return

    if what in ('function', 'method'):
        func = obj
    else:
        func = obj.__init__

    try:
        overloads = typing.get_overloads(func)
    except AttributeError:
        return

    if not overloads:
        return

    for i, line in enumerate(lines):
        if line.startswith(':Param') or line.startswith(':Return'):
            break
    else:
        return

    for overload in overloads:
        sig = inspect.signature(overload, eval_str=True)
        args = sig.format().replace('= Ellipsis', '= ...')
        args = re.sub(
            r'\bnetloader(?:\.[A-Za-z_]\w*)+\b',
            lambda m: m.group().split('.')[-1],
            args,
        )
        overload_lines.append(f'.. {what}:: {name.rsplit(".", maxsplit=1)[-1]}{args}')
        overload_lines.append('   :no-index:')
        overload_lines.append('   :module: None')

    overload_lines.append('')
    lines[i:i] = overload_lines


def process_docstring(
        _: Any,
        what: str,
        name: str,
        obj: type,
        __: Any,
        lines: list[str]) -> None:
    """
    Modifies the docstring of a function or method.

    Parameters
    ----------
    what : str
        The type of the object being documented
    name : str
        The fully qualified name of the object being documented
    obj : type
        The object being documented
    lines : list[str]
        The lines of the docstring to be modified
    """
    if what == 'method' and name.startswith('netloader.') and name.endswith('.to'):
        lines[:] = ['See :meth:`torch.nn.Module.to`.']

    overload_docstring(_, what, name, obj, __, lines)


OBJ_REF = {
    key: _numpydoc_module_convert(key, value) for key, value in NUMPYDOC_ALIASES.items()
}
