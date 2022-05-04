from collections import defaultdict
from itertools import chain
import os
from setuptools import setup, find_namespace_packages  # type: ignore


HEADLESS = bool(int(os.getenv("DIPCOATIMAGE_HEADLESS", 0)))


def get_package_name():
    if HEADLESS:
        name = "dipcoatimage-finitedepth-headless"
    else:
        name = "dipcoatimage-finitedepth"
    return name


VERSION_FILE = "dipcoatimage/finitedepth/version.py"


def get_version():
    with open(VERSION_FILE, "r") as f:
        exec(compile(f.read(), VERSION_FILE, "exec"))
    return locals()["__version__"]


HEADLESS_EXCLUDE = [
    "dipcoatimage/finitedepth_gui",
]

PACKAGE_DATA = [
    "dipcoatimage/finitedepth/samples",
    "dipcoatimage/finitedepth/py.typed",
    "dipcoatimage/finitedepth_gui/icons",
    "dipcoatimage/finitedepth_gui/py.typed",
]


def get_packages():
    packages = []

    EXCLUDE = []
    if HEADLESS:
        EXCLUDE += [path.replace("/", ".") for path in HEADLESS_EXCLUDE]
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


def get_install_requires():
    REQS = read_requirements("requirements/install.txt")
    # Add GUI dependencies if not headless
    if not HEADLESS:
        GUI_REQS = read_requirements("requirements/gui.txt")
        REQS.extend(GUI_REQS)
    return REQS


def get_extras_require():
    ret = {}

    if HEADLESS:
        ret["test"] = read_requirements("requirements/test.txt")
    else:
        ret["test"] = read_requirements("requirements/test.txt") + read_requirements(
            "requirements/test-gui.txt"
        )

    ret["test-ci"] = (
        read_requirements("requirements/test.txt")
        + read_requirements("requirements/test-gui.txt")
        + read_requirements("requirements/test-ci.txt")
    )

    ret["doc"] = read_requirements("requirements/doc.txt")

    ret["full"] = list(set(chain(*ret.values())))
    return ret


setup(
    name=get_package_name(),
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
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)
