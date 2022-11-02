"""Module to mock PySide6 objects."""

from PySide6.QtCore import Qt


TypeRole = Qt.UserRole
DataRole = Qt.UserRole + 1  # type: ignore[operator]
