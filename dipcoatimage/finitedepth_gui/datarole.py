"""
File to mock PySide6 without installing it in document building.
"""

from PySide6.QtCore import Qt


TypeRole = Qt.ItemDataRole.UserRole
DataRole = Qt.ItemDataRole.UserRole + 1
