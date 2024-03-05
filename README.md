# DipCoatImage-FiniteDepth

[![License](https://img.shields.io/github/license/dipcoat-image/finitedepth)](https://github.com/dipcoat-image/finitedepth/blob/master/LICENSE)
[![CI](https://github.com/dipcoat-image/finitedepth/actions/workflows/ci.yml/badge.svg)](https://github.com/dipcoat-image/finitedepth/actions/workflows/ci.yml)
[![CD](https://github.com/dipcoat-image/finitedepth/actions/workflows/cd.yml/badge.svg)](https://github.com/dipcoat-image/finitedepth/actions/workflows/cd.yml)
[![Docs](https://readthedocs.org/projects/dipcoatimage-finitedepth/badge/?version=latest)](https://dipcoatimage-finitedepth.readthedocs.io/en/latest/?badge=latest)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/dipcoatimage-finitedepth.svg)](https://pypi.python.org/pypi/dipcoatimage-finitedepth/)
[![PyPI Version](https://img.shields.io/pypi/v/dipcoatimage-finitedepth.svg)](https://pypi.python.org/pypi/dipcoatimage-finitedepth/)

![title](https://dipcoatimage-finitedepth.readthedocs.io/en/latest/_images/index-1.png)

DipCoatImage-FiniteDepth is a simple and extensible Python package to analyze dip coating with finite immersion depth.

## Usage

Store analysis parameters in configuration file (YAML or JSON).

```
data1:
 type: CoatingImage
 referencePath: ref.png
 targetPath: target.png
 output:
     layerData: output/data.csv
data2:
   type: MyType
   my-parameters: ...
data3:
   ...
```

Pass the file to command:

```
$ finitedepth analyze config.yml
```

You can also define your own analysis type by [writing a plugin](https://dipcoatimage-finitedepth.readthedocs.io/en/latest/plugin.html).

## Installation

DipCoatImage-FiniteDepth can be installed using `pip`.

```
$ pip install dipcoatimage-finitedepth
```

Optional dependencies are listed in [manual](https://dipcoatimage-finitedepth.readthedocs.io/en/latest/intro.html#installation).

## Documentation

DipCoatImage-FiniteDepth is documented with [Sphinx](https://pypi.org/project/Sphinx/).
The manual can be found on Read the Docs:

> https://dipcoatimage-finitedepth.readthedocs.io

If you want to build the document yourself, get the source code and install with `[doc]` dependency.
Then, go to `doc` directory and build the document:

```
$ pip install .[doc]
$ cd doc
$ make html
```

Document will be generated in `build/html` directory. Open `index.html` to see the central page.

## Citation

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
