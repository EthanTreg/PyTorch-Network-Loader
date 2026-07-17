"""
Sphinx configuration for PyTorch Network Loader.
"""
from __future__ import annotations

import sys
from typing import Any
from pathlib import Path

from sphinx.application import Sphinx

# Ensure NetLoader can be imported
DOCS_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = DOCS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from docs.utils import utils

# General configuration
nitpicky: bool = True
add_module_names: bool = False
release: str = '3.11.2'
author: str = 'Ethan Tregidga'
project: str = 'PyTorch Network Loader'
copyright = '2026, Ethan Tregidga'
templates_path: list[str] = ['_templates']
exclude_patterns: list[str] = ['_build', 'Thumbs.db', '.DS_Store']
extensions: list[str] = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_design',
]

# Autodoc configuration
autodoc_typehints: str = 'none'
autodoc_default_options: dict[str, Any] = {
    'members': True,
    'show-inheritance': True,
    'class-doc-from': 'both',
    'member-order': 'groupwise',
}

# NumPyDoc configuration
numpydoc_xref_rtype: bool = True
numpydoc_xref_param_type: bool = True
numpydoc_show_class_members: bool = False

# HTML configuration
html_use_index: bool = False
html_theme: str = 'shibuya'
html_static_path: list[str] = ['_static']
html_css_files: list[str] = ['custom.css']

# Cross-project links
intersphinx_mapping: dict[str, tuple[str, None]] = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'zuko': ('https://zuko.readthedocs.io/stable/', None),
}
numpydoc_xref_aliases = utils.NUMPYDOC_ALIASES


def setup(app: Sphinx) -> None:
    """
    Setup function for Sphinx configuration.

    Parameters
    ----------
    app : Sphinx
        Sphinx application instance
    """
    app.add_autodocumenter(utils.TypeDocumenter)
    app.connect('autodoc-process-docstring', utils.process_docstring)
    app.add_directive(utils.BaseAutoSignature.name, utils.BaseAutoSignature)
    app.add_directive(utils.AutoLossSignature.name, utils.AutoLossSignature)
    app.add_directive(utils.AutoLayerSignature.name, utils.AutoLayerSignature)
    app.add_directive(utils.AutoArchsSignature.name, utils.AutoArchsSignature)
