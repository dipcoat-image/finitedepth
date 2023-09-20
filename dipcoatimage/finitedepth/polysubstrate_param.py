import dataclasses


__all__ = [
    "Parameters",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """
    Parameters for :class:`PolySubstrate`.

    Parameters
    ----------
    Sigma: positive float
        Standard deviation of Gaussian kernel to smooth the noise.
    Rho: positive float
        Radian resolution for Hough transformation to detect the sidelines.
    Theta: positive float
        Angle resolution for Hough transformation to detect the sidelines.

    """

    Sigma: float
    Rho: float
    Theta: float
