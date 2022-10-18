"""
Main Window
===========

V2 for analysisgui.py
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QDockWidget
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from dipcoatimage.finitedepth_gui.views import ExperimentListView, DataViewTab


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
        self._dataViewTab = DataViewTab()

        self._listView.setModel(self._model)
        self._dataViewTab.setModel(self._model)

        exptListDock = QDockWidget("List of experiments")
        exptListDock.setWidget(self._listView)
        self.addDockWidget(Qt.LeftDockWidgetArea, exptListDock)
        exptDataDock = QDockWidget("Experiment data")
        exptDataDock.setWidget(self._dataViewTab)
        self.addDockWidget(Qt.BottomDockWidgetArea, exptDataDock)
        self.setWindowTitle("Coating layer analysis")