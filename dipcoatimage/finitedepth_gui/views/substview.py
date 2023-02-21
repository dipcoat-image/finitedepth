"""
Substrate view
==============

"""

import dataclasses
import dawiq
from PySide6.QtCore import Qt, Slot, QModelIndex
from PySide6.QtWidgets import (
    QWidget,
    QDataWidgetMapper,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
)
from dipcoatimage.finitedepth_gui.model import (
    ExperimentDataModel,
    IndexRole,
)
from .importview import ImportDataView, ImportArgsDelegate
from typing import Optional


__all__ = [
    "SubstrateView",
    "SubstrateArgsDelegate",
]


class SubstrateView(QWidget):
    """
    Widget to :class:`SubstrateArgs`.

    >>> from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import (
    ...     ExperimentDataListView,
    ...     SubstrateView,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentDataListView()
    ...     exptListWidget.setModel(model)
    ...     layout.addWidget(exptListWidget)
    ...     substWidget = SubstrateView()
    ...     substWidget.setModel(model)
    ...     layout.addWidget(substWidget)
    ...     window.setLayout(layout)
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None

        self._importView = ImportDataView()
        self._paramStackWidget = dawiq.DataclassStackedWidget()
        self._drawOptStackWidget = dawiq.DataclassStackedWidget()

        self._typeMapper = QDataWidgetMapper()
        self._argsMapper = QDataWidgetMapper()

        self._importView.editingFinished.connect(self._typeMapper.submit)
        self._typeMapper.setOrientation(Qt.Orientation.Vertical)
        self._typeMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._typeMapper.setItemDelegate(ImportArgsDelegate())
        self._paramStackWidget.currentDataEdited.connect(self._argsMapper.submit)
        self._drawOptStackWidget.currentDataEdited.connect(self._argsMapper.submit)
        self._argsMapper.setOrientation(Qt.Orientation.Vertical)
        self._argsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._argsMapper.setItemDelegate(SubstrateArgsDelegate())

        self._importView.setTitle("Substrate type")
        self._paramStackWidget.addWidget(
            QGroupBox("Parameters")  # default empty widget
        )
        self._drawOptStackWidget.addWidget(
            QGroupBox("Draw options")  # default empty widget
        )

        layout = QVBoxLayout()
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
        self._typeMapper.clearMapping()
        self._argsMapper.clearMapping()
        self._typeMapper.setModel(model)
        self._argsMapper.setModel(model)
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)
            self._typeMapper.addMapping(self._importView, model.Row_SubstArgs)
            self._argsMapper.addMapping(
                self._paramStackWidget, model.Row_SubstParameters
            )
            self._argsMapper.addMapping(
                self._drawOptStackWidget, model.Row_SubstDrawOptions
            )

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._typeMapper.setRootIndex(index)
            self._typeMapper.toFirst()
            substArgsIndex = model.getIndexFor(IndexRole.SUBSTARGS, index)
            self._argsMapper.setRootIndex(substArgsIndex)
            self._argsMapper.toFirst()
        else:
            self._typeMapper.setCurrentModelIndex(QModelIndex())
            self._argsMapper.setCurrentModelIndex(QModelIndex())
            self._importView.clear()
            self._paramStackWidget.setCurrentIndex(0)
            self._drawOptStackWidget.setCurrentIndex(0)


class SubstrateArgsDelegate(dawiq.DataclassDelegate):
    TypeRole = ExperimentDataModel.Role_DataclassType
    DataRole = ExperimentDataModel.Role_DataclassData

    def ignoreMissing(self) -> bool:
        return False

    def cacheModelData(cls, model, index, value, role):
        if isinstance(model, ExperimentDataModel):
            model.cacheData(index, value, role)
        else:
            super().cacheModelData(model, index, value, role)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel) and isinstance(
            editor, dawiq.DataclassStackedWidget
        ):
            # add data widget if absent
            dclsType = model.data(index, role=self.TypeRole)
            if isinstance(dclsType, type) and dataclasses.is_dataclass(dclsType):
                dclsIdx = editor.indexOfDataclass(dclsType)
                if dclsIdx == -1:
                    widget = dawiq.dataclass2Widget(dclsType)
                    indexRole = model.whatsThisIndex(index)
                    if indexRole == IndexRole.SUBST_PARAMETERS:
                        title = "Parameters"
                    elif indexRole == IndexRole.SUBST_DRAWOPTIONS:
                        title = "Draw options"
                    else:
                        title = ""
                    widget.setTitle(title)
                    dclsIdx = editor.addDataWidget(widget, dclsType)
            else:
                dclsIdx = 0
            editor.setCurrentIndex(dclsIdx)
            self.setEditorData(editor.currentWidget(), index)
        else:
            super().setEditorData(editor, index)
