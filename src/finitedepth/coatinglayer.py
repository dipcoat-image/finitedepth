"""Analyze coating layer."""

__all__ = [
    "CoatingLayer",
    "RectLayerShape",
]


class CoatingLayer:
    """Basic coating layer."""

    @classmethod
    def from_dict(cls, d):
        """Construct an instance from a dictionary *d*."""
        ...


class RectLayerShape:
    """Coating layer over rectangular substrate."""

    @classmethod
    def from_dict(cls, d):
        """Construct an instance from a dictionary *d*."""
        ...
