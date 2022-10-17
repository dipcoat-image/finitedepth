"""
Main Window
===========

V2 for analysisgui.py
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QTabWidget,
    QScrollArea,
)
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from dipcoatimage.finitedepth_gui.views import (
    ExperimentListView,
    ExperimentView,
    ReferenceView,
    SubstrateView,
    CoatingLayerView,
    AnalysisView,
)


__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """
    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui import MainWindow
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     app.setStyleSheet("*[requiresFieldData=true]{border: 1px solid red}")
    ...     window = MainWindow()
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = ExperimentDataModel()
        self._listView = ExperimentListView()
        self._exptView = ExperimentView()
        self._refView = ReferenceView()
        self._substView = SubstrateView()
        self._layerView = CoatingLayerView()
        self._analysisView = AnalysisView()

        self._listView.setModel(self._model)
        self._exptView.setModel(self._model)
        self._refView.setModel(self._model)
        self._substView.setModel(self._model)
        self._layerView.setModel(self._model)
        self._analysisView.setModel(self._model)

        exptListDock = QDockWidget("List of experiments")
        exptListDock.setWidget(self._listView)
        self.addDockWidget(Qt.LeftDockWidgetArea, exptListDock)
        exptDataDock = QDockWidget("Experiment data")
        exptDataTabWidget = QTabWidget()
        exptScroll = QScrollArea()
        exptScroll.setWidgetResizable(True)
        exptScroll.setWidget(self._exptView)
        exptDataTabWidget.addTab(exptScroll, "Experiment")
        refScroll = QScrollArea()
        refScroll.setWidgetResizable(True)
        refScroll.setWidget(self._refView)
        exptDataTabWidget.addTab(refScroll, "Reference")
        substScroll = QScrollArea()
        substScroll.setWidgetResizable(True)
        substScroll.setWidget(self._substView)
        exptDataTabWidget.addTab(substScroll, "Substrate")
        layerScroll = QScrollArea()
        layerScroll.setWidgetResizable(True)
        layerScroll.setWidget(self._layerView)
        exptDataTabWidget.addTab(layerScroll, "Coating layer")
        analysisScroll = QScrollArea()
        analysisScroll.setWidgetResizable(True)
        analysisScroll.setWidget(self._analysisView)
        exptDataTabWidget.addTab(analysisScroll, "Analysis")
        exptDataDock.setWidget(exptDataTabWidget)
        self.addDockWidget(Qt.BottomDockWidgetArea, exptDataDock)
        self.setWindowTitle("Coating layer analysis")
