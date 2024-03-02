"""Manage reference image and ROIs."""

__all__ = [
    "Reference",
]


class Reference:
    """Basic reference image."""

    @classmethod
    def from_dict(cls, d): ...
