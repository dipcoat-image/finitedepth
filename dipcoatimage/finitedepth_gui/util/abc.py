from abc import ABCMeta
from PySide6.QtCore import QObject


__all__ = [
    "AbstractObjectType",
]


class AbstractObjectType(type(QObject), ABCMeta):
    pass
