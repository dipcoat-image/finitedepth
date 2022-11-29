"""
Analysis view
=============

"""

import dawiq
import os
from PySide6.QtCore import Qt, Slot, QModelIndex
from PySide6.QtWidgets import (
    QWidget,
    QLineEdit,
    QComboBox,
    QDataWidgetMapper,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QHBoxLayout,
    QStyledItemDelegate,
)
from dipcoatimage.finitedepth import Analyzer, AnalysisArgs
from dipcoatimage.finitedepth_gui.worker import AnalysisState
from dipcoatimage.finitedepth_gui.model import (
    ExperimentDataModel,
    IndexRole,
)
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
    ...     ExperimentDataListView,
    ...     AnalysisView,
    ... )
    >>> def runGUI():
    ...     app = QApplication(sys.argv)
    ...     model = ExperimentDataModel()
    ...     window = QWidget()
    ...     layout = QHBoxLayout()
    ...     exptListWidget = ExperimentDataListView()
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

        self._analyzeArgsMapper = QDataWidgetMapper()

        self._dataPathLineEdit.editingFinished.connect(self._analyzeArgsMapper.submit)
        for ext in Analyzer.data_writers.keys():
            self._dataExtComboBox.addItem(f".{ext}")
        self._dataExtComboBox.activated.connect(self._analyzeArgsMapper.submit)
        self._imgPathLineEdit.editingFinished.connect(self._analyzeArgsMapper.submit)
        for ext in ["png", "jpg"]:
            self._imgExtComboBox.addItem(f".{ext}")
        self._imgExtComboBox.activated.connect(self._analyzeArgsMapper.submit)
        self._vidPathLineEdit.editingFinished.connect(self._analyzeArgsMapper.submit)
        for ext in Analyzer.video_codecs.keys():
            self._vidExtComboBox.addItem(f".{ext}")
        self._vidExtComboBox.activated.connect(self._analyzeArgsMapper.submit)
        self._fpsLineEdit.editingFinished.connect(self._analyzeArgsMapper.submit)
        self._fpsLineEdit.setValidator(dawiq.EmptyFloatValidator())
        self._analyzeButton.toggled.connect(self._onAnalyzeButtonToggle)
        self._analyzeArgsMapper.setOrientation(Qt.Orientation.Vertical)
        self._analyzeArgsMapper.setSubmitPolicy(QDataWidgetMapper.ManualSubmit)
        self._analyzeArgsMapper.setItemDelegate(AnalysisArgsDelegate())

        self._dataPathLineEdit.setPlaceholderText("Data file path")
        self._dataPathLineEdit.setToolTip("Path for quantitative analysis result.")
        self._imgPathLineEdit.setPlaceholderText("Image file path")
        self._imgPathLineEdit.setToolTip(
            "Path for visualized image file.\n"
            "For multiple images, pass paths with format (e.g. img_%02d.jpg)."
        )
        self._vidPathLineEdit.setPlaceholderText("Video file path")
        self._vidPathLineEdit.setToolTip("Path for visualized video file.")
        self._fpsLineEdit.setPlaceholderText("(Optional) image FPS")
        self._fpsLineEdit.setToolTip("FPS value for multi-image experiment.")
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
        oldModel = self.model()
        if oldModel is not None:
            oldModel.activatedIndexChanged.disconnect(self.setActivatedIndex)
            oldModel.analysisStateChanged.disconnect(self._onAnalysisStateChange)
            oldModel.analysisProgressMaximumChanged.disconnect(
                self._progressBar.setMaximum
            )
            oldModel.analysisProgressValueChanged.disconnect(self._progressBar.setValue)
        self._model = model
        self._analyzeArgsMapper.clearMapping()
        self._analyzeArgsMapper.setModel(model)
        if model is not None:
            model.activatedIndexChanged.connect(self.setActivatedIndex)
            model.analysisStateChanged.connect(self._onAnalysisStateChange)
            model.analysisProgressMaximumChanged.connect(self._progressBar.setMaximum)
            model.analysisProgressValueChanged.connect(self._progressBar.setValue)
            self._analyzeArgsMapper.addMapping(self, model.Row_AnalysisArgs)

    def dataPathName(self) -> str:
        return self._dataPathLineEdit.text()

    def setDataPathName(self, name: str):
        self._dataPathLineEdit.setText(name)

    def dataPathExtension(self) -> str:
        return self._dataExtComboBox.currentText()

    def setDataPathExtension(self, extension: str):
        self._dataExtComboBox.setCurrentText(extension)

    def imagePathName(self) -> str:
        return self._imgPathLineEdit.text()

    def setImagePathName(self, name: str):
        self._imgPathLineEdit.setText(name)

    def imagePathExtension(self) -> str:
        return self._imgExtComboBox.currentText()

    def setImagePathExtension(self, extension: str):
        self._imgExtComboBox.setCurrentText(extension)

    def videoPathName(self) -> str:
        return self._vidPathLineEdit.text()

    def setVideoPathName(self, name: str):
        self._vidPathLineEdit.setText(name)

    def videoPathExtension(self) -> str:
        return self._vidExtComboBox.currentText()

    def setVideoPathExtension(self, extension: str):
        self._vidExtComboBox.setCurrentText(extension)

    def fps(self) -> Optional[float]:
        fpsText = self._fpsLineEdit.text()
        return None if not fpsText else float(fpsText)

    def setFPS(self, fps: Optional[float]):
        fpsText = "" if fps is None else str(fps)
        self._fpsLineEdit.setText(fpsText)

    @Slot(QModelIndex)
    def setActivatedIndex(self, index: QModelIndex):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            self._analyzeArgsMapper.setRootIndex(index)
            self._analyzeArgsMapper.toFirst()
            self._analyzeButton.setCheckable(True)
        else:
            self._analyzeArgsMapper.setCurrentModelIndex(QModelIndex())
            self._dataPathLineEdit.clear()
            self._imgPathLineEdit.clear()
            self._vidPathLineEdit.clear()
            self._fpsLineEdit.clear()
            self._analyzeButton.setCheckable(False)

    def _onAnalyzeButtonToggle(self, checked: bool):
        model = self.model()
        if model is None:
            return
        index = model.activatedIndex()
        worker = model.worker(index)
        if worker is None:
            return
        if checked:
            state = AnalysisState.Running
        else:
            state = AnalysisState.Stopped
        worker.setAnalysisState(state)

    def _onAnalysisStateChange(self, state: AnalysisState):
        if state == AnalysisState.Running:
            self._analyzeButton.setChecked(True)
            self._analyzeButton.setText("Stop analysis")
        else:
            self._analyzeButton.setChecked(False)
            self._analyzeButton.setText("Analyze")


class AnalysisArgsDelegate(QStyledItemDelegate):
    def setModelData(self, editor, model, index):
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.ANALYSISARGS and isinstance(editor, AnalysisView):
                dataPathName = editor.dataPathName()
                dataPathExt = editor.dataPathExtension()
                if not dataPathName:
                    dataPath = ""
                else:
                    dataPath = dataPathName + dataPathExt

                imgPathName = editor.imagePathName()
                imgPathExt = editor.imagePathExtension()
                if not imgPathName:
                    imgPath = ""
                else:
                    imgPath = imgPathName + imgPathExt

                vidPathName = editor.videoPathName()
                vidPathExt = editor.videoPathExtension()
                if not vidPathName:
                    vidPath = ""
                else:
                    vidPath = vidPathName + vidPathExt

                fps = editor.fps()

                analysisArgs = AnalysisArgs(dataPath, imgPath, vidPath, fps)
                model.cacheData(index, analysisArgs, model.Role_AnalysisArgs)

        else:
            super().setModelData(editor, model, index)

    def setEditorData(self, editor, index):
        model = index.model()
        if isinstance(model, ExperimentDataModel):
            indexRole = model.whatsThisIndex(index)
            if indexRole == IndexRole.ANALYSISARGS and isinstance(editor, AnalysisView):
                analysisArgs = model.data(index, role=model.Role_AnalysisArgs)

                dataPathName, dataPathExt = os.path.splitext(analysisArgs.data_path)
                editor.setDataPathName(dataPathName)
                editor.setDataPathExtension(dataPathExt)

                imgPathName, imgPathExt = os.path.splitext(analysisArgs.image_path)
                editor.setImagePathName(imgPathName)
                editor.setImagePathExtension(imgPathExt)

                vidPathName, vidPathExt = os.path.splitext(analysisArgs.video_path)
                editor.setVideoPathName(vidPathName)
                editor.setVideoPathExtension(vidPathExt)

                editor.setFPS(analysisArgs.fps)

            super().setEditorData(editor, index)
