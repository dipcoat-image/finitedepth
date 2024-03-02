"""Analyze substrate geometry."""

__all__ = [
    "Substrate",
    "RectSubstrate",
]


class Substrate:
    """Basic substrate."""

    @classmethod
    def from_dict(cls, d): ...


class RectSubstrate:
    """Rectangular substrate."""

    @classmethod
    def from_dict(cls, d): ...
