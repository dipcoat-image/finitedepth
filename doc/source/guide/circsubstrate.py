import dataclasses

import cv2
import numpy as np

from dipcoatimage.finitedepth import ReferenceBase, SubstrateBase
from dipcoatimage.finitedepth.cache import attrcache
from dipcoatimage.finitedepth.substrate import SubstDrawOpt


@dataclasses.dataclass(frozen=True)
class CircParam:
    dp: float
    minDist: float
    param1: float
    param2: float


@dataclasses.dataclass(frozen=True)
class CircleData:
    r: float


class CircSubstrate(SubstrateBase[ReferenceBase, CircParam, SubstDrawOpt, CircleData]):
    ParamType = CircParam
    DrawOptType = SubstDrawOpt
    DataType = CircleData

    @attrcache("_hough")
    def houghCircles(self):
        return cv2.HoughCircles(
            self.image(),
            cv2.HOUGH_GRADIENT,
            self.parameters.dp,
            self.parameters.minDist,
            param1=self.parameters.param1,
            param2=self.parameters.param2,
        )

    def verify(self):
        assert self.houghCircles() is not None, "No circle detected"

    def region_points(self):
        x, y, r = np.round(self.houghCircles()[0, 0, :]).astype(np.int32)
        return np.array([[x, y]])

    def draw(self):
        img = cv2.cvtColor(self.image(), cv2.COLOR_GRAY2RGB)
        if self.houghCircles() is not None:
            x, y, r = np.round(self.houghCircles()[0, 0, :]).astype(np.uint16)
            cv2.circle(img, (x, y), r, (0, 255, 0), 3)
        return img

    def analyze(self):
        _, _, r = self.houghCircles()[0, 0, :]
        return self.DataType(r)
