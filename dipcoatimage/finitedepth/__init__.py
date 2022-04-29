"""
Package to analyze the coating layer shape in finite-depth dip coating process.
"""

from .version import __version__  # noqa

from .reference import (
    SubstrateReferenceBase,
    SubstrateReference,
)
from .util import (
    get_samples_path,
)


__all__ = [
    "SubstrateReferenceBase",
    "SubstrateReference",
    "get_samples_path",
]
