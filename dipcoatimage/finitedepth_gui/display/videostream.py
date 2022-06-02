import cv2  # type: ignore[import]
from cv2PySide6 import NDArrayVideoPlayer, ClickableSlider
from dipcoatimage.finitedepth_gui.core import ClassSelection
from dipcoatimage.finitedepth_gui.workers import MasterWorker
import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QUrl, Signal, Slot, Qt, QObject
from PySide6.QtWidgets import QWidget, QPushButton, QHBoxLayout, QStyle
from PySide6.QtMultimedia import QMediaPlayer
from typing import Optional


__all__ = [
    "MediaController",
    "PreviewableNDArrayVideoPlayer",
    "VisualizeProcessor",
]


class MediaController(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._slider = ClickableSlider()
        self._playButton = QPushButton()
        self._stopButton = QPushButton()
        self._player = None
        self._pausedBySliderPress = False

        self.playButton().clicked.connect(self.onPlayButtonClicked)
        self.stopButton().clicked.connect(self.onStopButtonClicked)
        self.slider().sliderPressed.connect(self.onSliderPress)
        self.slider().sliderMoved.connect(self.onSliderMove)
        self.slider().sliderReleased.connect(self.onSliderRelease)

        layout = QHBoxLayout()
        play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        self.playButton().setIcon(play_icon)
        layout.addWidget(self.playButton())
        stop_icon = self.style().standardIcon(QStyle.SP_MediaStop)
        self.stopButton().setIcon(stop_icon)
        layout.addWidget(self.stopButton())
        self.slider().setOrientation(Qt.Horizontal)
        layout.addWidget(self.slider())
        self.setLayout(layout)

    def slider(self) -> ClickableSlider:
        return self._slider

    def playButton(self) -> QPushButton:
        return self._playButton

    def stopButton(self) -> QPushButton:
        return self._stopButton

    def player(self) -> Optional[QMediaPlayer]:
        return self._player

    @Slot()
    def onPlayButtonClicked(self):
        if self.player() is not None:
            if self.player().playbackState() == QMediaPlayer.PlayingState:
                self.player().pause()
            else:
                self.player().play()

    @Slot()
    def onStopButtonClicked(self):
        if self.player() is not None:
            self.player().stop()

    @Slot()
    def onSliderPress(self):
        if (
            self.player() is not None
            and self.player().playbackState() == QMediaPlayer.PlayingState
        ):
            self._pausedBySliderPress = True
            self.player().pause()
            self.player().setPosition(self.slider().value())

    @Slot(int)
    def onSliderMove(self, position: int):
        player = self.player()
        if player is not None:
            player.setPosition(position)

    @Slot()
    def onSliderRelease(self):
        if self.player() is not None and self._pausedBySliderPress:
            self.player().play()
            self._pausedBySliderPress = False

    def setPlayer(self, player: Optional[QMediaPlayer]):
        old_player = self.player()
        if old_player is not None:
            self.disconnectPlayer(old_player)
        self._player = player
        if player is not None:
            self.connectPlayer(player)

    def connectPlayer(self, player: QMediaPlayer):
        player.durationChanged.connect(  # type: ignore[attr-defined]
            self.onMediaDurationChange
        )
        player.playbackStateChanged.connect(  # type: ignore[attr-defined]
            self.onPlaybackStateChange
        )
        player.positionChanged.connect(  # type: ignore[attr-defined]
            self.onMediaPositionChange
        )

    def disconnectPlayer(self, player: QMediaPlayer):
        player.durationChanged.disconnect(  # type: ignore[attr-defined]
            self.onMediaDurationChange
        )
        player.playbackStateChanged.disconnect(  # type: ignore[attr-defined]
            self.onPlaybackStateChange
        )
        player.positionChanged.disconnect(  # type: ignore[attr-defined]
            self.onMediaPositionChange
        )

    @Slot(int)
    def onMediaDurationChange(self, duration: int):
        self.slider().setRange(0, duration)

    @Slot(QMediaPlayer.PlaybackState)
    def onPlaybackStateChange(self, state: QMediaPlayer.PlaybackState):
        if state == QMediaPlayer.PlayingState:
            pause_icon = self.style().standardIcon(QStyle.SP_MediaPause)
            self.playButton().setIcon(pause_icon)
        else:
            play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
            self.playButton().setIcon(play_icon)

    @Slot(int)
    def onMediaPositionChange(self, position: int):
        self.slider().setValue(position)


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
        if path:
            cap = cv2.VideoCapture(path)
            ok, img = cap.read()
            cap.release()
            if not ok:
                img = np.empty((0, 0, 0))
        else:
            img = np.empty((0, 0, 0))
        return img


class VisualizeProcessor(QObject):

    arrayChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._selectedClass = ClassSelection.EXPERIMENT

    def visualizeWorker(self) -> Optional[MasterWorker]:
        return self._worker

    def selectedClass(self) -> ClassSelection:
        return self._selectedClass

    def setVisualizeWorker(self, worker: Optional[MasterWorker]):
        self._worker = worker

    def setSelectedClass(self, select: ClassSelection):
        self._selectedClass = select

    @Slot(np.ndarray)
    def setArray(self, array: npt.NDArray[np.uint8]):
        """
        Process *array* with :meth:`processArray` and emit to
        :attr:`arrayChanged`.
        """
        self.arrayChanged.emit(self.processArray(array))

    def processArray(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        # only the selected class is updated, e.g. updated reference instance is not
        # applied to substrate worker when visualizing reference.
        worker = self.visualizeWorker()
        if worker is None:
            ret = array
        elif self.selectedClass() == ClassSelection.REFERENCE:
            worker.referenceWorker().setImage(array)
            worker.referenceWorker().updateReference()
            ret = worker.referenceWorker().visualizedImage()
        elif self.selectedClass() == ClassSelection.SUBSTRATE:
            worker.referenceWorker().setImage(array)
            worker.referenceWorker().updateReference()
            worker.substrateWorker().setReference(worker.referenceWorker().reference())
            worker.substrateWorker().updateSubstrate()
            ret = worker.substrateWorker().visualizedImage()
        else:
            ret = worker.experimentWorker().visualizeImage(array)
        return ret

    def emitVisualizationFromModel(self, select: ClassSelection):
        worker = self.visualizeWorker()
        if worker is None:
            return
        if select == ClassSelection.REFERENCE:
            ret = worker.referenceWorker().visualizedImage()
            self.arrayChanged.emit(ret)
        elif select == ClassSelection.SUBSTRATE:
            ret = worker.substrateWorker().visualizedImage()
            self.arrayChanged.emit(ret)
