from PySide6.QtCore import QMetaObject, QUrl
from typing import Protocol, Any

__all__ = [
    "SignalProtocol",
    "VideoPlayerProtocol",
    "CameraProtocol",
    "ImageCaptureProtocol",
    "MediaRecorderProtocol",
]


class SignalProtocol(Protocol):
    def connect(self, *args, **kwargs) -> QMetaObject.Connection:
        ...

    def disconnect(self, *args, **kwargs) -> bool:
        ...

    def emit(self, *args, **kwargs):
        ...


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
    cameraDeviceChanged: SignalProtocol
    activeChanged: SignalProtocol

    def setCameraDevice(self, device: Any):
        ...

    def setActive(self, active: bool):
        ...


class ImageCaptureProtocol(Protocol):
    ...


class MediaRecorderProtocol(Protocol):
    ...
