"""
Main Window
===========

"""

import os
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QMainWindow, QDockWidget, QPushButton, QFileDialog
from dipcoatimage.finitedepth_gui.core import FrameSource
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from dipcoatimage.finitedepth_gui.views import ExperimentDataListView, DataViewTab
from dipcoatimage.finitedepth_gui.display import MainDisplayWindow
from dipcoatimage.finitedepth_gui.visualize import PySide6Visualizer


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
        self._visualizer = PySide6Visualizer()
        self._listView = ExperimentDataListView()
        self._dataViewTab = DataViewTab()
        self._display = MainDisplayWindow()
        self._cwdButton = QPushButton()

        self._listView.setModel(self._model)
        self._dataViewTab.setModel(self._model)
        self._visualizer.setModel(self._model)
        self._display.setModel(self._model)

        self._visualizer.roiMaximumChanged.connect(self._dataViewTab.setROIMaximum)
        self._visualizer.arrayChanged.connect(self._display.setArray)
        self._dataViewTab.currentViewChanged.connect(self._visualizer.setCurrentView)
        self._dataViewTab.currentViewChanged.connect(self._display.setCurrentView)
        self._dataViewTab.roiDrawModeChanged.connect(self._display.setROIDrawMode)
        self._display.setPlayer(self._visualizer.videoPlayer())
        self._display.setCamera(self._visualizer.camera())
        self._display.setImageCapture(self._visualizer.imageCapture())
        self._display.setMediaRecorder(self._visualizer.mediaRecorder())
        self._display.visualizationModeChanged.connect(
            self._visualizer.setVisualizationMode
        )
        self._cwdButton.clicked.connect(self.browseCWD)
        self.statusBar().addPermanentWidget(self._cwdButton)
        self.statusBar().showMessage(os.getcwd())

        exptListDock = QDockWidget("List of experiments")
        exptListDock.setWidget(self._listView)
        self.addDockWidget(Qt.LeftDockWidgetArea, exptListDock)
        exptDataDock = QDockWidget("Experiment data")
        exptDataDock.setWidget(self._dataViewTab)
        self.addDockWidget(Qt.BottomDockWidgetArea, exptDataDock)
        self.setCentralWidget(self._display)
        self.setWindowTitle("Coating layer analysis")
        self._cwdButton.setText("Browse")
        self._cwdButton.setToolTip("Change current directory")

        self._visualizer.setFrameSource(FrameSource.FILE)

    @Slot()
    def browseCWD(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select current working directory",
            "./",
            options=QFileDialog.Option.ShowDirsOnly
            | QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self.setCWD(path)

    def setCWD(self, path: str):
        os.chdir(path)
        self.statusBar().showMessage(path)

    def closeEvent(self, event):
        self._visualizer.stop()
        super().closeEvent(event)
