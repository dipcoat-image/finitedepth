# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DipCoatImage-FiniteDepth"
copyright = "2024, Jisoo Song"
author = "Jisoo Song"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_typehints = "description"

autoapi_dirs = ["../../src"]
autoapi_template_dir = "_templates/autoapi"
autoapi_root = "reference"


def autoapi_skip(app, what, name, obj, skip, options):
    if what == "module" and name in [
        "finitedepth.__main__",
    ]:
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", autoapi_skip)


intersphinx_mapping = {
    "python": ("http://docs.python.org/", None),
    "pip": ("https://pip.pypa.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mypy": ("https://mypy.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "DipCoatImage-FiniteDepth"
html_static_path = []

plot_html_show_formats = False
plot_html_show_source_link = False

# -- Custom scripts ----------------------------------------------------------

# Reference file

f = open("help-finitedepth.txt", "w")
subprocess.call(["finitedepth", "-h"], stdout=f)
f.close()

f = open("help-finitedepth-samples.txt", "w")
subprocess.call(["finitedepth", "samples", "-h"], stdout=f)
f.close()

f = open("help-finitedepth-analyze.txt", "w")
subprocess.call(["finitedepth", "analyze", "-h"], stdout=f)
f.close()
