"""
Testing helpers
===============

:mod:`dipcoatimage.finitedepth.util.testing` provides functions to help writing
unit tests.

"""

from collections.abc import Iterable
from typing import Dict


__all__ = ["dict_includes"]


def dict_includes(sup: Dict, sub: Dict):
    """Recursively check if *sup* is superset of *sub*."""
    for key, value in sub.items():
        if key not in sup:
            return False
        if isinstance(value, dict):
            if not dict_includes(sup[key], value):
                return False
        elif isinstance(value, Iterable):
            if not list(value) == list(sup[key]):
                return False
        else:
            if not value == sup[key]:
                return False
    return True
