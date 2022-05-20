import cv2  # type: ignore[import]
from cv2PySide6 import NDArrayVideoPlayer
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QUrl, Slot


__all__ = [
    "PreviewableNDArrayVideoPlayer",
]


class PreviewableNDArrayVideoPlayer(NDArrayVideoPlayer):
    """
    Video player which emits first frame of the video on source change
    and on video stop.
    """

    @Slot(QUrl)
    def setSource(self, source: QUrl):
        super().setSource(source)
        self.arrayChanged.emit(self.previewImage())

    @Slot()
    def stop(self):
        super().stop()
        self.arrayChanged.emit(self.previewImage())

    def previewImage(self) -> npt.NDArray[np.uint8]:
        path = self.source().toLocalFile()
        cap = cv2.VideoCapture(path)
        ok, img = cap.read()
        cap.release()
        if not ok:
            img = np.empty((0, 0, 0))
        return img
