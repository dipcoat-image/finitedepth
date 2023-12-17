# DipCoatImage-FiniteDepth

[![Build Status](https://github.com/dipcoat-image/finitedepth/actions/workflows/ci.yml/badge.svg)](https://github.com/dipcoat-image/finitedepth/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/dipcoatimage-finitedepth/badge/?version=latest)](https://dipcoatimage-finitedepth.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/dipcoat-image/finitedepth)](https://github.com/dipcoat-image/finitedepth/blob/master/LICENSE)

Python package for image analysis on finite depth dip coating process.

## Usage

Analysis can be done by passing configuration files.

```
$ finitedepth analyze config1.yml conf2.yml ...
```

## Installation

```
$ pip install git+https://github.com/dipcoat-image/finitedepth.git
```

## Documentation

Documentation can be found on Read the Docs:

> https://dipcoatimage-finitedepth.readthedocs.io/en/latest/

To build the document yourself, you must download the full source code of the project and install the package with `doc` dependency.

```
$ git clone https://github.com/dipcoat-image/finitedepth.git
$ cd finitedepth
$ pip install .[doc]
```

Then run the following command to build the document.

```
$ cd doc
$ make html
```

Documents will be generated in `doc/build/html` directory.
`index.html` file will lead you to main page.

# Citation

If you use this package in a scientific publication, please cite the following paper:

```
@article{song2023measuring,
  title={Measuring coating layer shape in arbitrary geometry},
  author={Song, Jisoo and Yu, Dongkeun and Jo, Euihyun and Nam, Jaewook},
  journal={Physics of Fluids},
  volume={35},
  number={12},
  year={2023},
  publisher={AIP Publishing}
}
```
