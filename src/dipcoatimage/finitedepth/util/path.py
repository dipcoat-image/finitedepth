"""
Path functions
==============

:mod:`dipcoatimage.finitedepth.util.path` provides functions to manage sample
paths in testing environment.

"""
from importlib_resources import files


__all__ = [
    "get_samples_path",
]


def get_samples_path(*paths: str) -> str:
    """
    Get the absolute path to the directory where the sample data are stored.

    Parameters
    ==========

    paths
        Subpaths under ``dipcoatimage/finitedepth/samples/`` directory.

    Returns
    =======

    path
        Absolute path to the sample depending on the user's system.

    Examples
    ========

    >>> from dipcoatimage.finitedepth import get_samples_path
    >>> get_samples_path() # doctest: +SKIP
    'path/dipcoatimage/finitedepth/samples'
    >>> get_samples_path("coat1.png") # doctest: +SKIP
    'path/dipcoatimage/finitedepth/samples/coat1.png'

    """
    data_path = files("dipcoatimage.finitedepth.data")
    if not paths:
        return str(data_path._paths[0])
    return str(data_path.joinpath(*paths))
