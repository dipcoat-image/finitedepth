import dataclasses
from typing import Optional, List


__all__ = [
    "CSVWriter",
    "Parameters",
]


async def CSVWriter(filename: str, header: List[str]):
    ...


@dataclasses.dataclass(frozen=True)
class Parameters:
    """
    Parameters for :class:`Analysis`.

    Attributes
    ----------
    ref_data, ref_visual : str
        Paths for data file and visualized file of reference image.
    subst_data, subst_visual : str
        Paths for data file and visualized file of substrate image.
    layer_data, layer_visual : str
        Paths for data file and visualized file of coating layer image(s).
        Pass formattable string (e.g. `img_%02d.jpg`) to save multiple images.
    layer_fps : float or None
        FPS to determine timestamps of coating layer data.
    """

    ref_data: str = ""
    ref_visual: str = ""
    subst_data: str = ""
    subst_visual: str = ""
    layer_data: str = ""
    layer_visual: str = ""
    layer_fps: Optional[float] = None
