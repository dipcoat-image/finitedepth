"""Analyze coating layer."""

__all__ = [
    "CoatingLayer",
    "RectLayerShape",
]


class CoatingLayer:
    """Basic coating layer."""

    @classmethod
    def from_dict(cls, d): ...


class RectLayerShape:
    """Coating layer over rectangular substrate."""

    @classmethod
    def from_dict(cls, d): ...
