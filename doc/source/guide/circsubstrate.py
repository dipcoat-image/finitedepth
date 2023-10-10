import cv2
import numpy as np

from dipcoatimage.finitedepth import ReferenceBase, SubstrateBase
from dipcoatimage.finitedepth.substrate import SubstData, SubstDrawOpt, SubstParam


class CircSubstrate(SubstrateBase[ReferenceBase, SubstParam, SubstDrawOpt, SubstData]):
    ParamType = SubstParam
    DrawOptType = SubstDrawOpt
    DataType = SubstData

    def region_points(self):
        return np.array([[self.image().shape[1] / 2, 0]], dtype=np.int32)

    def verify(self):
        pass

    def draw(self):
        return cv2.cvtColor(self.image(), cv2.COLOR_GRAY2RGB)

    def analyze(self):
        return self.DataType()
