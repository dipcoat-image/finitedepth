"""
Experiment
==========

:mod:`dipcoatimage.finitedepth.experiment` helps analyze the full dip coating
experiment with finite-depth substrate.

Single experiment consists of:

    * Bare substrate image file
    * Coated substrate image file(s), or video file

This module provides factory to handle consecutive images of coated substrate
with single image of bare substrate. Also, it serializes the parameters to
analyze the images and to save the result.

Analysis factory
----------------

Data serialization
------------------

.. autoclass:: ImportArgs
   :members:

.. autoclass:: ReferenceArgs
   :members:

"""

import cv2  # type: ignore
import dataclasses

from .reference import SubstrateReferenceBase
from .util import import_variable, data_converter, OptionalROI


__all__ = [
    "ImportArgs",
    "ReferenceArgs",
]


@dataclasses.dataclass
class ImportArgs:
    """Arguments to import the variable from module."""

    name: str = ""
    module: str = "dipcoatimage.finitedepth"


@dataclasses.dataclass
class ReferenceArgs:
    """
    Data for the concrete instance of :class:`SubstrateReferenceBase`.

    Parameters
    ==========

    type
        Information to import reference class.
        Class name defaults to ``SubstrateReference``.

    path
        Path to the reference image file.

    templateROI, substrateROI, parameters, draw_options
        Arguments for :class:`SubstrateReferenceBase` instance.

    Examples
    ========

    .. plot::
       :include-source:

       >>> import matplotlib.pyplot as plt #doctest: +SKIP
       >>> from dipcoatimage.finitedepth import get_samples_path
       >>> from dipcoatimage.finitedepth.experiment import ReferenceArgs
       >>> from dipcoatimage.finitedepth.util import cwd
       >>> refargs = ReferenceArgs(path='ref1.png',
       ...                         templateROI=(200, 100, 1200, 500),
       ...                         substrateROI=(300, 50, 1100, 600))
       >>> with cwd(get_samples_path()):
       ...     ref = refargs.as_reference()
       >>> plt.imshow(ref.draw()) #doctest: +SKIP

    """

    type: ImportArgs = dataclasses.field(default_factory=ImportArgs)
    path: str = ""
    templateROI: OptionalROI = (0, 0, None, None)
    substrateROI: OptionalROI = (0, 0, None, None)
    parameters: dict = dataclasses.field(default_factory=dict)
    draw_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.type.name:
            self.type.name = "SubstrateReference"

    def as_reference(self) -> SubstrateReferenceBase:
        """Construct the substrate reference instance."""
        # XXX: perhaps make this a static method of Experiment
        name = self.type.name
        module = self.type.module
        refcls = import_variable(name, module)
        if not (
            isinstance(refcls, type) and issubclass(refcls, SubstrateReferenceBase)
        ):
            raise TypeError(f"{refcls} is not substrate reference class.")

        img = cv2.imread(self.path)
        if img is None:
            raise TypeError(f"Invalid path: {self.path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        params = data_converter.structure(
            self.parameters, refcls.Parameters  # type: ignore
        )
        drawopts = data_converter.structure(
            self.draw_options, refcls.DrawOptions  # type: ignore
        )

        ref = refcls(  # type: ignore
            img,
            self.templateROI,
            self.substrateROI,
            parameters=params,
            draw_options=drawopts,
        )
        return ref
