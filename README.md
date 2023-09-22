# DipCoatImage-FiniteDepth

[![Build Status](https://github.com/JSS95/dawiq/actions/workflows/ci.yml/badge.svg)](https://github.com/dipcoat-image/finitedepth/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/dipcoatimage-finitedepth/badge/?version=latest)](https://dipcoatimage-finitedepth.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/dipcoat-image/finitedepth)](https://github.com/dipcoat-image/finitedepth/blob/master/LICENSE)

Python package for image analysis on finite depth dip coating process.

## Introduction

DipCoatImage-FiniteDepth is a Python package to perform image analysis on finite depth dip coating process.
It is a component of DipCoatImage, a collective package for image analysis on dip coating process.

## Installation

It is recommended to use a dedicated environment for DipCoatImage packages, which can be easily done by using [Anaconda](https://www.anaconda.com/).
The following command creates and acivates a virtual environment `dipcoatimage`, with `pip` installed.

```
$ conda create -n dipcoat-image pip
$ conda activate dipcoat-image
```

Now, install the package using `pip`.
The following command installs ``dipcoatimage-finitedepth`` directly from the repository.

```
$ pip install git+ssh://git@github.com/dipcoat-image/finitedepth.git
```

To install with additional options, refer to [Installation](https://github.com/dipcoat-image/finitedepth/blob/master/doc/source/user-guide/installation.rst) document.

# Documentation

Documentation of this project is done by [Sphinx](https://www.sphinx-doc.org/en/master/).
To build the document yourself, you must download the full source code of the project and install the package with `doc` dependency.

Run the following command on this path to build the document in HTML format.

```
$ cd doc
$ make html
```

HTML documents will be generated in doc/build/html directory. The main page is index.html file.
