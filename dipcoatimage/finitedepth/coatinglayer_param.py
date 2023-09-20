import enum

__all__ = [
    "SubtractionMode",
]


class SubtractionMode(enum.Flag):
    """
    Option to determine how the template matching result will be displayed.

    Template matching result is shown by subtracting the pixels from the
    background.

    Members
    -------
    NONE
        Do not show the template matching result.
    TEMPLATE
        Subtract the template ROI.
    SUBSTRRATE
        Subtract the substrate ROI.
    FULL
        Subtract both template and substrate ROIs.

    """

    NONE = 0
    TEMPLATE = 1
    SUBSTRATE = 2
    FULL = TEMPLATE | SUBSTRATE
