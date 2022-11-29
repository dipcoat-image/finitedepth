from PySide6.QtCore import QUrl
from typing import Protocol

__all__ = [
    "VideoPlayerProtocol",
    "CameraProtocol",
    "ImageCaptureProtocol",
    "MediaRecorderProtocol",
]


class VideoPlayerProtocol(Protocol):
    def play(self):
        ...

    def pause(self):
        ...

    def stop(self):
        ...

    def setSource(self, url: QUrl):
        ...


class CameraProtocol(Protocol):
    ...


class ImageCaptureProtocol(Protocol):
    ...


class MediaRecorderProtocol(Protocol):
    ...
