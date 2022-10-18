"""
Data view tab
=============

V2 for controlwidgets/controlwidget.py
"""

from PySide6.QtWidgets import QTabWidget, QScrollArea
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from .exptview import ExperimentView
from .refview import ReferenceView
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
    ...     ExperimentListView,
    ...     DataViewTab,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentListView()
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

    def __init__(self, parent=None):
        super().__init__(parent)

        self._exptView = ExperimentView()
        self._refView = ReferenceView()
        self._substView = SubstrateView()
        self._layerView = CoatingLayerView()
        self._analysisView = AnalysisView()

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
