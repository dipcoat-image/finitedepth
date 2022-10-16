"""
Analysis view
=============

V2 for controlwidgets/analysiswidget.py
"""

import dawiq
from PySide6.QtWidgets import (
    QWidget,
    QLineEdit,
    QComboBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QHBoxLayout,
    QStyledItemDelegate,
)
from dipcoatimage.finitedepth.analysis import Analyzer
from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
from typing import Optional

__all__ = [
    "AnalysisView",
    "AnalysisArgsDelegate",
]


class AnalysisView(QWidget):
    """
    Widget to display :class:`AnalysisArgs` and progress bar.

    >>> from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout
    >>> import sys
    >>> from dipcoatimage.finitedepth_gui.model import ExperimentDataModel
    >>> from dipcoatimage.finitedepth_gui.views import (
    ...     ExperimentListView,
    ...     AnalysisView,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentListView()
    ...     exptListWidget.setModel(model)
    ...     layout.addWidget(exptListWidget)
    ...     analysisWidget = AnalysisView()
    ...     analysisWidget.setModel(model)
    ...     layout.addWidget(analysisWidget)
    ...     window.setLayout(layout)
    ...     window.show()
    ...     app.exec()
    ...     app.quit()
    >>> runGUI() # doctest: +SKIP

    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._model = None
        self._dataPathLineEdit = QLineEdit()
        self._dataExtComboBox = QComboBox()
        self._imgPathLineEdit = QLineEdit()
        self._imgExtComboBox = QComboBox()
        self._vidPathLineEdit = QLineEdit()
        self._vidExtComboBox = QComboBox()
        self._fpsLineEdit = QLineEdit()
        self._analyzeButton = QPushButton()
        self._progressBar = QProgressBar()

        self._fpsLineEdit.setValidator(dawiq.EmptyFloatValidator())

        self._dataPathLineEdit.setPlaceholderText("Data file path")
        for ext in Analyzer.data_writers.keys():
            self._dataExtComboBox.addItem(f".{ext}")
        self._imgPathLineEdit.setPlaceholderText("Image file path")
        self._imgPathLineEdit.setToolTip(
            "Pass paths with format (e.g. img_%02d.jpg) to save multiple images."
        )
        for ext in ["png", "jpg"]:
            self._imgExtComboBox.addItem(f".{ext}")
        self._vidPathLineEdit.setPlaceholderText("Video file path")
        for ext in Analyzer.video_codecs.keys():
            self._vidExtComboBox.addItem(f".{ext}")
        self._fpsLineEdit.setPlaceholderText("(Optional) fps for image experiment")
        self._fpsLineEdit.setToolTip(
            "Set FPS value for analysis data and video of image experiment."
        )
        self._analyzeButton.setText("Analyze")

        layout = QVBoxLayout()
        dataPathLayOut = QHBoxLayout()
        dataPathLayOut.addWidget(self._dataPathLineEdit)
        dataPathLayOut.addWidget(self._dataExtComboBox)
        layout.addLayout(dataPathLayOut)
        imgPathLayOut = QHBoxLayout()
        imgPathLayOut.addWidget(self._imgPathLineEdit)
        imgPathLayOut.addWidget(self._imgExtComboBox)
        layout.addLayout(imgPathLayOut)
        vidPathLayOut = QHBoxLayout()
        vidPathLayOut.addWidget(self._vidPathLineEdit)
        vidPathLayOut.addWidget(self._vidExtComboBox)
        layout.addLayout(vidPathLayOut)
        layout.addWidget(self._fpsLineEdit)
        layout.addWidget(self._analyzeButton)
        layout.addWidget(self._progressBar)
        self.setLayout(layout)

    def model(self) -> Optional[ExperimentDataModel]:
        return self._model

    def setModel(self, model: Optional[ExperimentDataModel]):
        ...


class AnalysisArgsDelegate(QStyledItemDelegate):
    ...
