from typing import Protocol


__all__ = ["SignalProtocol"]


class SignalProtocol(Protocol):
    def connect(self, *args, **kwargs):
        ...

    def disconnect(self, *args, **kwargs):
        ...
