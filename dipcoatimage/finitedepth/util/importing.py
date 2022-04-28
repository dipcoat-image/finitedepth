"""
Dynamic importing
=================

:mod:`dipcoatimage.finitedepth.util.importing` provides functions to dynamically
import objects.

"""

import importlib


__all__ = [
    "import_variable",
    "get_importinfo",
]


def import_variable(name: str, module_name: str = "") -> object:
    """
    Import the variable from the module name.

    If *module_name* is not given, tries ``eval(name)``. Else, import the
    variable from that module.

    Parameters
    ==========

    name
        Name of the variable.

    module
        Name of the module that the variable will be imported from.

    Returns
    =======

    ret
        Any object that can be imported from the module.

    """
    if not name:
        raise TypeError("Empty variable name")

    SENTINEL = object()
    ret = SENTINEL

    if module_name:
        module = importlib.import_module(module_name)
        ret = getattr(module, name, SENTINEL)

    if ret is SENTINEL:
        # fallback to default eval
        ret = eval(name)

    return ret


def get_importinfo(var: object) -> str:
    """
    Get information to import the variable.

    Parameters
    ==========

    var
        Any object

    Returns
    =======

    modname
        Name of the module where *var* can be imported.

    """
    modname = getattr(var, "__module__", type(var).__module__)
    return modname
