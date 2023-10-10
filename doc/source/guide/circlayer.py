import dataclasses

import cv2
import numpy as np
from circsubstrate import CircSubst

from dipcoatimage.finitedepth import CoatingLayerBase
from dipcoatimage.finitedepth.cache import attrcache
from dipcoatimage.finitedepth.coatinglayer import LayerDecoOpt, LayerDrawOpt, LayerParam


@dataclasses.dataclass(frozen=True)
class Data:
    maxThickness: float


class CircLayer(
    CoatingLayerBase[CircSubst, LayerParam, LayerDrawOpt, LayerDecoOpt, Data]
):
    ParamType = LayerParam
    DrawOptType = LayerDrawOpt
    DecoOptType = LayerDecoOpt
    DataType = Data

    @attrcache("_maxdist")
    def max_dist(self):
        c = self.substrate_point() + self.substrate.hough_circles()[0, 0, :2]
        (layer_cnt,), _ = cv2.findContours(
            self.extract_layer().astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        dists = np.linalg.norm(layer_cnt - c, axis=-1)
        if dists.size == 0:
            return np.float64(0), np.empty((0, 2), np.int32)
        maxloc = np.argmax(dists)
        return dists[maxloc, 0], layer_cnt[maxloc]

    def verify(self):
        assert self.max_dist[0] > 0, "No coating layer"

    def draw(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        v = self.substrate_point().astype(np.int32)
        x, y, r = self.substrate.hough_circles()[0, 0, :].astype(np.int32)
        cv2.circle(img, (x, y) + v, r, (0, 0, 255), 3)
        if self.max_dist()[0] > 0:
            c = v + self.substrate.hough_circles()[0, 0, :2]
            p = self.max_dist()[1]
            cv2.line(img, c.astype(np.int32), p[0], (0, 255, 0), 3)
        return img

    def analyze(self):
        r = self.substrate.hough_circles()[0, 0, 2]
        return self.DataType(self.max_dist()[0] - r)
