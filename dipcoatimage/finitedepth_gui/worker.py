"""
Analysis worker
===============

V2 for workers
"""

from PySide6.QtCore import QRunnable


__all__ = [
    "ExperimentWorker",
]


class ExperimentWorker(QRunnable):
    ...
