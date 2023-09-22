"""
Data path
=========

:mod:`dipcoatimage.finitedepth.util.path` provides access to data files at
runtime.

"""
from importlib_resources import files


__all__ = [
    "get_data_path",
]


def get_data_path(*paths: str) -> str:
    """
    Get path to data file.

    Parameters
    ----------
    paths : str
        Subpaths under ``dipcoatimage/finitedepth/data/`` directory.

    Returns
    -------
    path
        Absolute path to the data.

    Examples
    ========

    >>> from dipcoatimage.finitedepth import get_data_path
    >>> get_data_path() # doctest: +SKIP
    'path/dipcoatimage/finitedepth/data'
    >>> get_data_path("coat1.png") # doctest: +SKIP
    'path/dipcoatimage/finitedepth/data/coat1.png'

    """
    data_path = files("dipcoatimage.finitedepth.data")
    if not paths:
        return str(data_path._paths[0])
    return str(data_path.joinpath(*paths))
