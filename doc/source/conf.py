# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os

import cv2
import numpy as np
import yaml

from dipcoatimage.finitedepth import Config, data_converter, get_data_path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

os.environ["FINITEDEPTH_DATA"] = get_data_path()


# -- Project information -----------------------------------------------------

project = "DipCoatImage-FiniteDepth"
copyright = "2022, Jisoo Song"
author = "Jisoo Song"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "autoapi.extension",
    "sphinx_tabs.tabs",
]

autodoc_member_order = "bysource"

autodoc_default_options = {
    "show-inheritance": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []  # type: ignore

intersphinx_mapping = {
    "python": ("http://docs.python.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "cattrs": ("https://cattrs.readthedocs.io/en/latest/", None),
}

numpydoc_show_class_members = False

autoapi_python_use_implicit_namespaces = True
autoapi_dirs = ["../../src/dipcoatimage/finitedepth"]
autoapi_template_dir = "_templates/autoapi"
autoapi_root = "reference"


def skip_submodules(app, what, name, obj, skip, options):
    if what == "module" and name in [
        "finitedepth.version",
        "finitedepth.__main__",
    ]:
        skip = True
    return skip


def setup(sphinx):
    sphinx.connect("autoapi-skip-member", skip_submodules)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "DipCoatImage-FiniteDepth"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/logo.png"

plot_html_show_formats = False
plot_html_show_source_link = False

# -- Custom scripts ----------------------------------------------------------

# Draw logo

data = dict(
    ref_path=get_data_path("ref3.png"),
    coat_path=get_data_path("coat3.mp4"),
    reference=dict(
        templateROI=(13, 10, 1246, 200),
        substrateROI=(100, 100, 1200, 500),
    ),
    coatinglayer=dict(
        deco_options=dict(
            layer=dict(
                facecolor=(69, 132, 182),
                linewidth=0,
            )
        )
    ),
)
config = data_converter.structure(data, Config)
coat = config.construct_coatinglayer(0, False)
img = coat.draw()

img[np.where(img == (0, 0, 0))[:-1]] = (100, 100, 100)

mask = coat.image.astype(bool) ^ coat.extract_layer()
k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
edge_mask = cv2.erode(mask.astype(np.uint8), k).astype(bool) ^ mask
img[edge_mask] = (255, 255, 255)

alpha = ~np.all(img == (255, 255, 255), axis=-1) * 255

os.makedirs("_static", exist_ok=True)
cv2.imwrite(
    "_static/logo.png",
    cv2.cvtColor(np.dstack([img, alpha]).astype(np.uint8), cv2.COLOR_RGBA2BGRA),
)

# Tutorial file

with open("tutorial/config-rect.yml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
(v,) = data.values()
config = data_converter.structure(v, Config)
config.analysis.parameters["subst_data"] = "tutorial/output/subst3.csv"
config.analysis.parameters["layer_visual"] = ""
config.analysis.parameters["layer_data"] = "tutorial/output/result3.csv"
config.analyze("Generating tutorial data...")
