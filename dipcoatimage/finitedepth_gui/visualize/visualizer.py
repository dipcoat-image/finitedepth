from araviq6 import FrameToArrayConverter
import numpy as np
from dipcoatimage.finitedepth_gui.core import FrameSource
from PySide6.QtCore import Slot
from PySide6.QtMultimedia import (
    QCamera,
    QMediaPlayer,
    QMediaCaptureSession,
    QVideoSink,
    QImageCapture,
    QMediaRecorder,
)
from .base import VisualizerBase


__all__ = [
    "PySide6Visualizer",
]


class PySide6Visualizer(VisualizerBase):
    """Visualizer using :mod:`PySide6.QtMultimedia`."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._videoPlayer = QMediaPlayer()
        self._playerSink = QVideoSink()
        self._lastVideoFrame = np.empty((0, 0, 0), dtype=np.uint8)

        self._camera = QCamera()
        self._captureSession = QMediaCaptureSession()
        self._imageCapture = QImageCapture()
        self._mediaRecorder = QMediaRecorder()
        self._cameraSink = QVideoSink()
        self._arrayConverter = FrameToArrayConverter()

        self._videoPlayer.setVideoSink(self._playerSink)

        self._camera.activeChanged.connect(self._onCameraActiveChange)
        self._captureSession.setCamera(self._camera)
        self._captureSession.setImageCapture(self._imageCapture)
        self._captureSession.setRecorder(self._mediaRecorder)
        self._captureSession.setVideoSink(self._cameraSink)

    def videoPlayer(self) -> QMediaPlayer:
        return self._videoPlayer

    def videoPlaybackState(self):
        state = self._videoPlayer.playbackState()
        if state == QMediaPlayer.PlaybackState.StoppedState:
            ret = self.PlaybackState.StoppedState
        elif state == QMediaPlayer.PlaybackState.PlayingState:
            ret = self.PlaybackState.PlayingState
        elif state == QMediaPlayer.PlaybackState.PausedState:
            ret = self.PlaybackState.PausedState
        else:
            raise ValueError(f"Unknown playback state: {state}")
        return ret

    def camera(self):
        return self._camera

    def imageCapture(self):
        return self._imageCapture

    def mediaRecorder(self):
        return self._mediaRecorder

    @Slot(bool)
    def _onCameraActiveChange(self, active: bool):
        if active:
            frameSource = FrameSource.CAMERA
        else:
            frameSource = FrameSource.FILE
        self.setFrameSource(frameSource)

    def togglePlayerPipeline(self, toggle: bool):
        if toggle:
            self._playerSink.videoFrameChanged.connect(
                self._arrayConverter.setVideoFrame
            )
            self._arrayConverter.arrayChanged.connect(self.visualizeImageFromPlayer)
        else:
            self._playerSink.videoFrameChanged.disconnect(
                self._arrayConverter.setVideoFrame
            )
            self._arrayConverter.arrayChanged.disconnect(self.visualizeImageFromPlayer)

    def toggleCameraPipeline(self, toggle: bool):
        if toggle:
            self._cameraSink.videoFrameChanged.connect(
                self._arrayConverter.setVideoFrame
            )
            self._arrayConverter.arrayChanged.connect(self.visualizeImageFromCamera)
        else:
            self._cameraSink.videoFrameChanged.disconnect(
                self._arrayConverter.setVideoFrame
            )
            self._arrayConverter.arrayChanged.disconnect(self.visualizeImageFromCamera)
