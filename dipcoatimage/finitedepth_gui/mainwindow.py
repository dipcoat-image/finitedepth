"""
Main Window
===========

V2 for analysisgui.py
"""

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt
from PySide6.QtWidgets import QMainWindow, QDockWidget
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from dipcoatimage.finitedepth_gui.views import ExperimentDataListView, DataViewTab
from dipcoatimage.finitedepth_gui.display import MainDisplayWindow_V2


__all__ = [
    "VisualizeProcessor",
    "MainWindow",
]


class VisualizeProcessor(QObject):
    arrayChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ready = True

    @Slot(np.ndarray)
    def setArray(self, array: npt.NDArray[np.uint8]):
        array = array.copy()  # must detach array from the memory
        self._ready = False
        self.arrayChanged.emit(self.processArray(array))
        self._ready = True

    def processArray(self, array: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        # TODO: implement processing
        return array

    def ready(self) -> bool:
        return self._ready


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
        self._listView = ExperimentDataListView()
        self._dataViewTab = DataViewTab()
        self._display = MainDisplayWindow_V2()

        self._processorThread = QThread()
        self._imageProcessor = VisualizeProcessor()

        self._listView.setModel(self._model)
        self._dataViewTab.setModel(self._model)
        self._display.setModel(self._model)
        self._display.setImageProcessor(self._imageProcessor)

        self._imageProcessor.moveToThread(self._processorThread)
        self._processorThread.start()

        exptListDock = QDockWidget("List of experiments")
        exptListDock.setWidget(self._listView)
        self.addDockWidget(Qt.LeftDockWidgetArea, exptListDock)
        exptDataDock = QDockWidget("Experiment data")
        exptDataDock.setWidget(self._dataViewTab)
        self.addDockWidget(Qt.BottomDockWidgetArea, exptDataDock)
        self.setCentralWidget(self._display)
        self.setWindowTitle("Coating layer analysis")

    def closeEvent(self, event):
        self._processorThread.quit()
        self._processorThread.wait()
        super().closeEvent(event)
