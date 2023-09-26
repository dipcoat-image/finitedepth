import csv
import dataclasses
import imageio.v2 as iio  # TODO: use PyAV
import mimetypes
from typing import Optional, List


__all__ = [
    "ImageWriter",
    "CSVWriter",
    "Parameters",
]


def ImageWriter(path: str, fps: Optional[float] = None):
    mtype, _ = mimetypes.guess_type(path)
    if mtype is None:
        raise TypeError(f"Unsupported mimetype: {mtype}.")

    ftype, subtype = mtype.split("/")
    try:
        path % 0
        formattable = True
    except (TypeError, ValueError):
        formattable = False
    if ftype == "video" or subtype in [
        "gif",
        "tiff",
    ]:
        writer = iio.get_writer(path, fps=fps)
        try:
            while True:
                img = yield
                writer.append_data(img)
        finally:
            writer.close()
    elif ftype == "image":
        i = 0
        while True:
            img = yield
            if formattable:
                path = path % i
            iio.imwrite(path, img)
            i += 1
    else:
        raise TypeError(f"Unsupported mimetype: {mtype}.")


def CSVWriter(path: str, header: List[str]):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        while True:
            data = yield
            writer.writerow(data)


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
