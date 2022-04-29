"""
Package to analyze the coating layer shape in finite-depth dip coating process.
"""

from .version import __version__  # noqa

from .reference import (
    SubstrateReferenceBase,
)
from .util import (
    get_samples_path,
)


__all__ = [
    "SubstrateReferenceBase",
    "get_samples_path",
]
