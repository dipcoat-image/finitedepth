from typing import Protocol

__all__ = [
    "VideoPlayerProtocol",
    "CameraProtocol",
    "CaptureSessionProtocol",
    "ImageCaptureProtocol",
    "MediaRecorderProtocol",
    "VideoSinkProtocol",
]


class VideoPlayerProtocol(Protocol):
    ...


class CameraProtocol(Protocol):
    ...


class CaptureSessionProtocol(Protocol):
    ...


class ImageCaptureProtocol(Protocol):
    ...


class MediaRecorderProtocol(Protocol):
    ...


class VideoSinkProtocol(Protocol):
    ...
