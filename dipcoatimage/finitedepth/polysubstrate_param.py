import dataclasses


__all__ = [
    "Parameters",
]


@dataclasses.dataclass(frozen=True)
class Parameters:
    """
    Parameters for :class:`PolySubstrate`.

    Attributes
    ----------
    Sigma: positive float
        Standard deviation of Gaussian kernel to smooth the noise.
    Rho: positive float
        Radian resolution for Hough transformation to detect the sidelines.
    Theta: positive float
        Angle resolution for Hough transformation to detect the sidelines.
    Step: positive int
        Sampling rate of points for Hough transformation. Larger step makes
        evaluation faster.

    """

    Sigma: float
    Rho: float
    Theta: float
    Step: int = 1
