"""
Data view tab
=============

V2 for controlwidgets/controlwidget.py
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTabWidget, QScrollArea
from dipcoatimage.finitedepth_gui.core import TypeFlag
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from .exptview import ExperimentView
from .refview import ReferenceView, ROIDrawFlag
from .substview import SubstrateView
from .layerview import CoatingLayerView
from .analysisview import AnalysisView
from typing import Optional


__all__ = [
    "DataViewTab",
]


class DataViewTab(QTabWidget):
    """
    Tab widget for data views:

    * :class:`ExperimentView`
    * :class:`ReferenceView`
    * :class:`SubstrateView`
    * :class:`CoatingLayerView`
    * :class:`AnalysisView`

    >>> from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import (
    ...     ExperimentDataListView,
    ...     DataViewTab,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentDataListView()
    ...     exptListWidget.setModel(model)
    ...     layout.addWidget(exptListWidget)
    ...     tab = DataViewTab()
    ...     tab.setModel(model)
    ...     layout.addWidget(tab)
    ...     window.setLayout(layout)
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    currentTypeChanged = Signal(TypeFlag)
    roiDrawFlagChanged = Signal(ROIDrawFlag)
    analysisRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._exptView = ExperimentView()
        self._refView = ReferenceView()
        self._substView = SubstrateView()
        self._layerView = CoatingLayerView()
        self._analysisView = AnalysisView()

        self.currentChanged.connect(self._onCurrentChange)
        self._refView.roiDrawFlagChanged.connect(self.roiDrawFlagChanged)
        self._analysisView.analysisRequested.connect(self.analysisRequested)

        exptScroll = QScrollArea()
        exptScroll.setWidgetResizable(True)
        exptScroll.setWidget(self._exptView)
        self.addTab(exptScroll, "Experiment")
        refScroll = QScrollArea()
        refScroll.setWidgetResizable(True)
        refScroll.setWidget(self._refView)
        self.addTab(refScroll, "Reference")
        substScroll = QScrollArea()
        substScroll.setWidgetResizable(True)
        substScroll.setWidget(self._substView)
        self.addTab(substScroll, "Substrate")
        layerScroll = QScrollArea()
        layerScroll.setWidgetResizable(True)
        layerScroll.setWidget(self._layerView)
        self.addTab(layerScroll, "Coating layer")
        analysisScroll = QScrollArea()
        analysisScroll.setWidgetResizable(True)
        analysisScroll.setWidget(self._analysisView)
        self.addTab(analysisScroll, "Analysis")

    def setModel(self, model: Optional[ExperimentDataModel]):
        self._exptView.setModel(model)
        self._refView.setModel(model)
        self._substView.setModel(model)
        self._layerView.setModel(model)
        self._analysisView.setModel(model)

    def _onCurrentChange(self, index):
        self._refView._tempROIDrawButton.setChecked(False)
        self._refView._substROIDrawButton.setChecked(False)
        self.roiDrawFlagChanged.emit(ROIDrawFlag.NONE)

        widget = self.widget(index)
        if not isinstance(widget, QScrollArea):
            typeFlag = TypeFlag.UNKNOWN
        elif widget.widget() is self._exptView:
            typeFlag = TypeFlag.EXPERIMENT
        elif widget.widget() is self._refView:
            typeFlag = TypeFlag.REFERENCE
        elif widget.widget() is self._substView:
            typeFlag = TypeFlag.SUBSTRATE
        elif widget.widget() is self._layerView:
            typeFlag = TypeFlag.COATINGLAYER
        elif widget.widget() is self._analysisView:
            typeFlag = TypeFlag.ANALYSIS
        else:
            typeFlag = TypeFlag.UNKNOWN
        self.currentTypeChanged.emit(typeFlag)
