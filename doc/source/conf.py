# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess

import cv2
import numpy as np

from finitedepth import CoatingLayer, Reference, Substrate, get_sample_path

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
        "finitedepth.cache",
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
html_static_path = ["_static"]
html_logo = "_static/logo.png"

plot_html_show_formats = False
plot_html_show_source_link = False

# -- Custom scripts ----------------------------------------------------------

# Draw logo

img = cv2.imread(get_sample_path("ref.png"), cv2.IMREAD_GRAYSCALE)
_, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ref = Reference(bin, (10, 10, 1250, 200), (100, 100, 1200, 500))
subst = Substrate(ref)
img = cv2.imread(get_sample_path("coat.png"), cv2.IMREAD_GRAYSCALE)
_, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
coat = CoatingLayer(bin, subst)

logo = coat.draw(layer_color=(69, 132, 182), layer_thickness=-1)
logo[np.where(logo == (0, 0, 0))[:-1]] = (100, 100, 100)
mask = coat.image.astype(bool) ^ coat.extract_layer()
k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
edge_mask = cv2.erode(mask.astype(np.uint8), k).astype(bool) ^ mask
logo[edge_mask] = (255, 255, 255)
alpha = ~np.all(logo == (255, 255, 255), axis=-1) * 255

os.makedirs("_static", exist_ok=True)
cv2.imwrite(
    "_static/logo.png",
    cv2.cvtColor(np.dstack([logo, alpha]).astype(np.uint8), cv2.COLOR_RGBA2BGRA),
)

# Reference file

f = open("help-finitedepth.txt", "w")
subprocess.call(["finitedepth", "-h"], stdout=f)
f.close()

f = open("help-finitedepth-samples.txt", "w")
subprocess.call(["finitedepth", "samples", "-h"], stdout=f)
f.close()

f = open("help-finitedepth-references.txt", "w")
subprocess.call(["finitedepth", "references", "-h"], stdout=f)
f.close()

f = open("help-finitedepth-substrates.txt", "w")
subprocess.call(["finitedepth", "substrates", "-h"], stdout=f)
f.close()

f = open("help-finitedepth-coatinglayers.txt", "w")
subprocess.call(["finitedepth", "coatinglayers", "-h"], stdout=f)
f.close()

f = open("help-finitedepth-analyze.txt", "w")
subprocess.call(["finitedepth", "analyze", "-h"], stdout=f)
f.close()
