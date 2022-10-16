"""
Reference view
==============

V2 for controlwidgets/refwidget.py
"""

import dawiq
from PySide6.QtCore import Slot, QModelIndex
from PySide6.QtWidgets import (
    QWidget,
    QLineEdit,
    QPushButton,
    QDataWidgetMapper,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
)
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from .importview import ImportDataView
from typing import Optional


__all__ = [
    "ReferenceView",
    "ReferenceArgsDelegate",
]


class ReferenceView(QWidget):
    """
    Widget to display reference file path and :class:`ReferenceArgs`.

    >>> from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import (
    ...     ExperimentListView,
    ...     ReferenceView,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentListView()
    ...     exptListWidget.setModel(model)
    ...     layout.addWidget(exptListWidget)
    ...     refWidget = ReferenceView()
    ...     refWidget.setModel(model)
    ...     layout.addWidget(refWidget)
    ...     window.setLayout(layout)
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._refPathLineEdit = QLineEdit()
        self._refPathMapper = QDataWidgetMapper()
        self._browseButton = QPushButton()
        self._importView = ImportDataView()
        self._paramStackWidget = dawiq.DataclassStackedWidget()
        self._drawOptStackWidget = dawiq.DataclassStackedWidget()
        self._refArgsDelegate = ReferenceArgsDelegate()
        self._refArgsMapper = QDataWidgetMapper()

        self._refPathLineEdit.setPlaceholderText("Path for the reference image file")
        self._browseButton.setText("Browse")
        self._importView.setTitle("Reference type")
        self._paramStackWidget.addWidget(
            QGroupBox("Parameters")  # default empty widget
        )
        self._drawOptStackWidget.addWidget(
            QGroupBox("Draw options")  # default empty widget
        )

        layout = QVBoxLayout()
        pathLayout = QHBoxLayout()
        pathLayout.addWidget(self._refPathLineEdit)
        pathLayout.addWidget(self._browseButton)
        layout.addLayout(pathLayout)
        layout.addWidget(self._importView)
        dataLayout = QHBoxLayout()
        dataLayout.addWidget(self._paramStackWidget)
        dataLayout.addWidget(self._drawOptStackWidget)
        layout.addLayout(dataLayout)
        self.setLayout(layout)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self.setActivatedIndex)
        self._model = model
        self._refPathMapper.setModel(model)
        self._refPathMapper.addMapping(self._refPathLineEdit, 0)
        self._refArgsMapper.setModel(model)
        self._refArgsMapper.addMapping(self, 0)
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._refPathMapper.setRootIndex(index)
            refPathIndex = model.index(model.ROW_REFPATH, 0, index)
            self._refPathMapper.setCurrentModelIndex(refPathIndex)
            self._refArgsMapper.setRootIndex(index)
            refIndex = model.index(model.ROW_REFERENCE, 0, index)
            self._refArgsMapper.setCurrentModelIndex(refIndex)
        else:
            self._refPathLineEdit.clear()
            self._importView.clear()
            self._refPathMapper.setCurrentModelIndex(QModelIndex())
            self._paramStackWidget.setCurrentIndex(0)
            self._drawOptStackWidget.setCurrentIndex(0)
            self._refArgsMapper.setCurrentModelIndex(QModelIndex())


class ReferenceArgsDelegate(dawiq.DataclassDelegate):
    def ignoreMissing(self) -> bool:
        return False
