from collections import defaultdict
from itertools import chain
import os
from setuptools import setup, find_namespace_packages  # type: ignore


VERSION_FILE = "dipcoatimage/finitedepth/version.py"


def get_version():
    with open(VERSION_FILE, "r") as f:
        exec(compile(f.read(), VERSION_FILE, "exec"))
    return locals()["__version__"]


PACKAGE_DATA = [
    "dipcoatimage/finitedepth/samples",
    "dipcoatimage/finitedepth/py.typed",
]


def get_packages():
    packages = []

    EXCLUDE = []
    # package data are included to package_data argument
    EXCLUDE += [path.replace("/", ".") for path in PACKAGE_DATA]

    for pkg in find_namespace_packages(include=["dipcoatimage.*"]):
        if not any(pkg.startswith(exclude) for exclude in EXCLUDE):
            packages.append(pkg)
    return packages


def get_package_data():
    ret = defaultdict(list)
    for path in PACKAGE_DATA:
        split = path.split("/")
        pkg = ".".join(split[:2])
        subpath = "/".join(split[2:])
        if os.path.isdir(path):
            subpath += "/*"
        elif os.path.isfile(path):
            pass
        else:
            continue
        ret[pkg].append(subpath)
    return dict(ret)


def read_requirements(path):
    with open(path, "r") as f:
        ret = f.read().splitlines()
    return ret


def get_extras_require():
    ret = {}

    ret["test"] = read_requirements("requirements/test.txt")

    ret["doc"] = read_requirements("requirements/doc.txt")

    ret["full"] = list(set(chain(*ret.values())))
    return ret


setup(
    name="dipcoatimage-finitedepth",
    version=get_version(),
    python_requires=">=3.9",
    description=(
        "Python package for image analysis on finite depth dip coating process"
    ),
    author="Jisoo Song",
    author_email="jeesoo9595@snu.ac.kr",
    url="https://github.com/dipcoat-image/finitedepth",
    packages=get_packages(),
    package_data=get_package_data(),
    install_requires=read_requirements("requirements/install.txt"),
    extras_require=get_extras_require(),
)
