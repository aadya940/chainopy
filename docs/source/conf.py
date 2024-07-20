# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Root Path
sys.path.insert(0, os.path.abspath("../../."))


project = "Chainopy"
copyright = "2024, Aadya A. Chinubhai"
author = "Aadya A. Chinubhai"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.extlinks",
    "sphinx_autodoc_typehints",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = [
    ".ipynb_checkpoints",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "chainopy.tests.rst",
    "chainopy._backend.rst",
    "chainopy._caching.rst",
    "chainopy._fileio.rst",
    "chainopy._visualization.rst",
    "chainopy._exceptions.rst",
    "modules.rst",
    "setup.rst",
    "_templates",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__init__",
    "private-members": False,
}

autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


def skip_cython_docstring(app, what, name, obj, skip, options):
    # Skip docstrings in Cython files
    if what == "module" and name.endswith(".pyx"):
        return True
    return skip


def setup(app):
    # Call event.
    app.connect("autodoc-skip-member", skip_cython_docstring)
