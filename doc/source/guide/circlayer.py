import dataclasses

import cv2
from circsubstrate import CircSubst

from dipcoatimage.finitedepth import CoatingLayerBase
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

    def verify(self):
        pass

    def draw(self):
        return cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

    def analyze(self):
        return self.Data(-1.0)
