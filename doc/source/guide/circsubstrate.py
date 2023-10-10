import dataclasses

import cv2
import numpy as np

from dipcoatimage.finitedepth import ReferenceBase, SubstrateBase
from dipcoatimage.finitedepth.cache import attrcache
from dipcoatimage.finitedepth.substrate import SubstDrawOpt


@dataclasses.dataclass(frozen=True)
class Param:
    dp: float
    minDist: float
    param1: float
    param2: float


@dataclasses.dataclass(frozen=True)
class Data:
    r: float


class CircSubst(SubstrateBase[ReferenceBase, Param, SubstDrawOpt, Data]):
    ParamType = Param
    DrawOptType = SubstDrawOpt
    DataType = Data

    @attrcache("_hough")
    def hough_circles(self):
        return cv2.HoughCircles(
            self.image(),
            cv2.HOUGH_GRADIENT,
            self.parameters.dp,
            self.parameters.minDist,
            param1=self.parameters.param1,
            param2=self.parameters.param2,
        )

    def verify(self):
        assert self.hough_circles() is not None, "No circle detected"

    def region_points(self):
        x, y, _ = np.round(self.hough_circles()[0, 0, :]).astype(np.int32)
        return np.array([[x, y]])

    def draw(self):
        img = cv2.cvtColor(self.image(), cv2.COLOR_GRAY2RGB)
        if self.hough_circles() is not None:
            x, y, r = self.hough_circles()[0, 0, :].astype(np.int32)
            cv2.circle(img, (x, y), r, (0, 255, 0), 3)
        return img

    def analyze(self):
        return self.DataType(self.hough_circles()[0, 0, 2])
