"""
Experiment view
===============

V2 for controlwidgets/exptwidget.py
"""

from PySide6.QtCore import Slot, QModelIndex
from PySide6.QtWidgets import (
    QWidget,
    QLineEdit,
    QListView,
    QDataWidgetMapper,
    QVBoxLayout,
    QGroupBox,
)
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from typing import Optional


__all__ = [
    "ExperimentWidget",
]


class ExperimentWidget(QWidget):
    """
    Widget to display experiment name, coating layer file paths and
    :class:`ExperimentArgs`.

    >>> from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import (
    ...     ExperimentListWidget,
    ...     ExperimentWidget
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentListWidget()
    ...     exptListWidget.setModel(model)
    ...     layout.addWidget(exptListWidget)
    ...     exptWidget = ExperimentWidget()
    ...     exptWidget.setModel(model)
    ...     layout.addWidget(exptWidget)
    ...     window.setLayout(layout)
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._nameLineEdit = QLineEdit()
        self._pathsListView = QListView()
        self._mapper = QDataWidgetMapper()

        layout = QVBoxLayout()
        layout.addWidget(self._nameLineEdit)
        pathsGroupBox = QGroupBox("Coating layer files path")
        pathsLayout = QVBoxLayout()
        pathsLayout.addWidget(self._pathsListView)
        pathsGroupBox.setLayout(pathsLayout)
        layout.addWidget(pathsGroupBox)
        self.setLayout(layout)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self.setActivatedIndex)
        self._model = model
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        self._pathsListView.setModel(model)
        self._pathsListView.setRootIndex(model.index(model.ROW_COATPATHS, 0, index))
