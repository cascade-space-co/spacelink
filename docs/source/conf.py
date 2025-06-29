import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "src",
        )
    ),
)

# Add the _ext directory to the path for custom extensions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "_ext")))

# -- Project information -----------------------------------------------------
project = "SpaceLink"
copyright = "2025, Cascade Space"
author = "Cascade Space"
release = "0.1.0"
version = release
root_doc = "index"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",  # Add MathJax support for LaTeX math
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "renku"

# -- Extension configuration -------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
