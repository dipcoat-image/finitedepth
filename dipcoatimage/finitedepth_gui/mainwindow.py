"""
Main Window
===========

V2 for analysisgui.py
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QDockWidget
from PySide6.QtMultimedia import QCamera
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from dipcoatimage.finitedepth_gui.views import ExperimentDataListView, DataViewTab
from dipcoatimage.finitedepth_gui.display import MainDisplayWindow_V2
from dipcoatimage.finitedepth_gui.pipeline import VisualizeManager


__all__ = [
    "MainWindow",
]


class MainWindow(QMainWindow):
    """
    >>> from PySide6.QtWidgets import QApplication
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui import MainWindow
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     app.setStyleSheet(
    ...     "*[requiresFieldValue=true]{border: 1px solid red}"
    ...     )
    ...     window = MainWindow()
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = ExperimentDataModel()
        self._visualizeManager = VisualizeManager()
        self._listView = ExperimentDataListView()
        self._dataViewTab = DataViewTab()
        self._display = MainDisplayWindow_V2()
        self._camera = QCamera()

        self._listView.setModel(self._model)
        self._dataViewTab.setModel(self._model)
        self._display.setModel(self._model)
        self._visualizeManager.setModel(self._model)
        self._visualizeManager.setCamera(self._camera)
        self._visualizeManager.setDisplay(self._display)
        self._display.setCamera(self._camera)

        exptListDock = QDockWidget("List of experiments")
        exptListDock.setWidget(self._listView)
        self.addDockWidget(Qt.LeftDockWidgetArea, exptListDock)
        exptDataDock = QDockWidget("Experiment data")
        exptDataDock.setWidget(self._dataViewTab)
        self.addDockWidget(Qt.BottomDockWidgetArea, exptDataDock)
        self.setCentralWidget(self._display)
        self.setWindowTitle("Coating layer analysis")

    def closeEvent(self, event):
        self._visualizeManager.stop()
        super().closeEvent(event)
