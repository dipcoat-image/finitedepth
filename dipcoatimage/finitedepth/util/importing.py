"""
Dynamic importing
=================

:mod:`dipcoatimage.finitedepth.util.importing` provides functions to dynamically
import objects.

"""

import enum
import importlib
from typing import Tuple, Any


__all__ = [
    "import_variable",
    "ImportStatus",
    "Importer",
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
        raise ValueError("Empty variable name")

    SENTINEL = object()
    if module_name:
        module = importlib.import_module(module_name)
        ret = getattr(module, name, SENTINEL)
        if ret is SENTINEL:
            raise ImportError(f"cannot import name {name} from {module_name}")
    else:
        ret = eval(name)

    return ret


class ImportStatus(enum.Enum):
    """
    Constants for variable import status of :class:`Importer`.

    0. VALID
        Variable is successfully imported.
    1. NO_MODULE
        Module is not found.
    2. NO_VARIABLE
        Module is found, but variable name is not importable from it.
    """

    VALID = 0
    NO_MODULE = 1
    NO_VARIABLE = 2


class Importer:
    """Class to try import the variable from variable name and module name."""
    __slots__ = ("varname", "modname")

    INVALID = object()

    def __init__(self, varname: str, modname: str = ""):
        self.varname = varname
        self.modname = modname

    def try_import(self) -> Tuple[Any, ImportStatus]:
        """
        Try import the variable with :attr:`varname` and :attr:`modname`.

        Returns
        =======

        (var, status)
            *var* is the imported object. If importing fails, :attr:`INVALID` is
            used as sentinel. *status* indicates the import status.
        """
        try:
            var = import_variable(self.varname, self.modname)
            status = ImportStatus.VALID
        except ModuleNotFoundError:
            var = self.INVALID
            status = ImportStatus.NO_MODULE
        except (ImportError, NameError, ValueError):
            var = self.INVALID
            status = ImportStatus.NO_VARIABLE
        return (var, status)
