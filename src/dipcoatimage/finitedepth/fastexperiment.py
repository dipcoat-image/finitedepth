import cv2
from .experiment import ExperimentBase
from .fastexperiment_param import Parameters


__all__ = [
    "FastExperiment",
]


class FastExperiment(ExperimentBase[Parameters]):
    """Simplest experiment class with no parameter."""

    __slots__ = ("_prev",)

    Parameters = Parameters

    def examine(self) -> None:
        return None

    def coatinglayer(
        self,
        image,
        substrate,
        *,
        layer_type,
        layer_parameters=None,
        layer_drawoptions=None,
        layer_decooptions=None,
    ):
        prev = getattr(self, "_prev", None)
        window = self.parameters.window
        if not prev or any(i < 0 for i in window):
            x0, y0, x1, y1 = substrate.reference.templateROI
        else:
            x0, y0, x1, y1 = substrate.reference.templateROI
            X, Y = prev
            w0, h0 = window
            w1, h1 = x1 - x0, y1 - y0
            H, W = image.shape[:2]
            X0, X1 = max(X - w0, 0), min(X + w1 + w0, W)
            Y0, Y1 = max(Y - h0, 0), min(Y + h1 + h0, H)
            image = image[Y0:Y1, X0:X1]

        template = substrate.reference.image[y0:y1, x0:x1]
        res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
        score, _, loc, _ = cv2.minMaxLoc(res)
        ret = layer_type(
            image,
            substrate,
            parameters=layer_parameters,
            draw_options=layer_drawoptions,
            deco_options=layer_decooptions,
            tempmatch=(loc, score),
        )
        self._prev = loc
        return ret
