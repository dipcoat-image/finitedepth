from dipcoatimage.finitedepth.analysis import AnalysisArgs, Analyzer
import os
from PySide6.QtCore import Slot
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QComboBox,
    QProgressBar,
)
from typing import List
from .base import ControlWidget


__all__ = [
    "EmptyDoubleValidator",
    "AnalysisWidget",
]


class EmptyDoubleValidator(QDoubleValidator):
    """Validator which accpets float and empty string"""

    def validate(self, input: str, pos: int) -> QDoubleValidator.State:
        ret = super().validate(input, pos)
        if not input:
            ret = QDoubleValidator.Acceptable
        return ret  # type: ignore[return-value]


class AnalysisWidget(ControlWidget):
    """
    Widget to analyze the experiment.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._blockModelUpdate = False

        self._datapath_lineedit = QLineEdit()
        self._data_ext_combobox = QComboBox()
        self._imgpath_lineedit = QLineEdit()
        self._img_ext_combobox = QComboBox()
        self._vidpath_lineedit = QLineEdit()
        self._vid_ext_combobox = QComboBox()
        self._imgexpt_fps_lineedit = QLineEdit()
        self._analyze_button = QPushButton()
        self._progressbar = QProgressBar()

        self.dataPathLineEdit().editingFinished.connect(self.commitToCurrentItem)
        self.dataExtensionComboBox().currentTextChanged.connect(
            self.commitToCurrentItem
        )
        self.imagePathLineEdit().editingFinished.connect(self.commitToCurrentItem)
        self.imageExtensionComboBox().currentTextChanged.connect(
            self.commitToCurrentItem
        )
        self.videoPathLineEdit().editingFinished.connect(self.commitToCurrentItem)
        self.videoExtensionComboBox().currentTextChanged.connect(
            self.commitToCurrentItem
        )
        self.imageFPSLineEdit().editingFinished.connect(self.commitToCurrentItem)
        self.imageFPSLineEdit().setValidator(EmptyDoubleValidator())

        self.dataPathLineEdit().setPlaceholderText("Data file path")
        self.imagePathLineEdit().setPlaceholderText("Image file path")
        self.imagePathLineEdit().setToolTip(
            "Pass paths with format (e.g. img_%02d.jpg) to save " "multiple images."
        )
        self.videoPathLineEdit().setPlaceholderText("Video file path")
        self.imageFPSLineEdit().setPlaceholderText(
            "(Optional) fps for image experiment"
        )
        self.imageFPSLineEdit().setToolTip(
            "Set FPS value for analysis data and video of image experiment."
        )
        self.analyzeButton().setText("Analyze")

        for ext in Analyzer.data_writers.keys():
            self.dataExtensionComboBox().addItem(f".{ext}")
        for ext in ["png", "jpg"]:
            self.imageExtensionComboBox().addItem(f".{ext}")
        for ext in Analyzer.video_codecs.keys():
            self.videoExtensionComboBox().addItem(f".{ext}")

        datapath_layout = QHBoxLayout()
        datapath_layout.addWidget(self.dataPathLineEdit())
        datapath_layout.addWidget(self.dataExtensionComboBox())
        imgpath_layout = QHBoxLayout()
        imgpath_layout.addWidget(self.imagePathLineEdit())
        imgpath_layout.addWidget(self.imageExtensionComboBox())
        vidpath_layout = QHBoxLayout()
        vidpath_layout.addWidget(self.videoPathLineEdit())
        vidpath_layout.addWidget(self.videoExtensionComboBox())
        anal_opt_layout = QVBoxLayout()
        anal_opt_layout.addLayout(datapath_layout)
        anal_opt_layout.addLayout(imgpath_layout)
        anal_opt_layout.addLayout(vidpath_layout)
        anal_opt_layout.addWidget(self.imageFPSLineEdit())
        anal_path_groupbox = QGroupBox("Analysis options")
        anal_path_groupbox.setLayout(anal_opt_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(anal_path_groupbox)
        main_layout.addWidget(self.analyzeButton())
        main_layout.addWidget(self.progressBar())
        self.setLayout(main_layout)

    def dataPathLineEdit(self) -> QLineEdit:
        """Line edit for data file path (without extesion)."""
        return self._datapath_lineedit

    def dataExtensionComboBox(self) -> QComboBox:
        """Combo box for data file extension."""
        return self._data_ext_combobox

    def imagePathLineEdit(self) -> QLineEdit:
        """Line edit for image file path (without extension)."""
        return self._imgpath_lineedit

    def imageExtensionComboBox(self) -> QComboBox:
        """Combo box for image file extension."""
        return self._img_ext_combobox

    def videoPathLineEdit(self) -> QLineEdit:
        """Line edit for video file path (without extension)."""
        return self._vidpath_lineedit

    def videoExtensionComboBox(self) -> QComboBox:
        """Combo box for video file extension."""
        return self._vid_ext_combobox

    def imageFPSLineEdit(self) -> QLineEdit:
        """Line edit for FPS of multi-image experiment."""
        return self._imgexpt_fps_lineedit

    def analyzeButton(self) -> QPushButton:
        """Button to trigger analysis."""
        return self._analyze_button

    def progressBar(self) -> QProgressBar:
        """Progress bar to display analysis progress."""
        return self._progressbar

    def setCurrentExperimentRow(self, row: int):
        super().setCurrentExperimentRow(row)

        self._blockModelUpdate = True
        model = self.experimentItemModel()
        args = model.data(model.index(row, model.Col_Analysis), model.Role_Args)
        data_path, data_ext = os.path.splitext(args.data_path)
        self.dataPathLineEdit().setText(data_path)
        self.dataExtensionComboBox().setCurrentText(data_ext)
        image_path, image_ext = os.path.splitext(args.image_path)
        self.imagePathLineEdit().setText(image_path)
        self.imageExtensionComboBox().setCurrentText(image_ext)
        video_path, video_ext = os.path.splitext(args.video_path)
        self.videoPathLineEdit().setText(video_path)
        self.videoExtensionComboBox().setCurrentText(video_ext)
        fps = str() if args.fps is None else str(args.fps)
        self.imageFPSLineEdit().setText(fps)
        self._blockModelUpdate = False

    def analysisArgs(self) -> AnalysisArgs:
        """Return :class:`analysisArgs` from current widget values."""
        data_path = (
            self.dataPathLineEdit().text() + self.dataExtensionComboBox().currentText()
        )
        image_path = (
            self.imagePathLineEdit().text()
            + self.imageExtensionComboBox().currentText()
        )
        video_path = (
            self.videoPathLineEdit().text()
            + self.videoExtensionComboBox().currentText()
        )
        fps_text = self.imageFPSLineEdit().text()
        fps = None if not fps_text else float(fps_text)
        args = AnalysisArgs(data_path, image_path, video_path, fps)
        return args

    @Slot()
    def commitToCurrentItem(self):
        """
        Set :meth:`analysisArgs` to currently activated item from
        :meth:`experimentItemModel`.
        """
        if not self._blockModelUpdate:
            model = self.experimentItemModel()
            row = self.currentExperimentRow()
            index = model.index(row, model.Col_Analysis)
            if index.isValid():
                model.setData(
                    index,
                    self.analysisArgs(),
                    model.Role_Args,  # type: ignore[arg-type]
                )

    def onExperimentsRemove(self, rows: List[int]):
        if self.currentExperimentRow() in rows:
            self._currentExperimentRow = -1
            self.dataPathLineEdit().clear()
            self.dataExtensionComboBox().setCurrentIndex(-1)
            self.imagePathLineEdit().clear()
            self.imageExtensionComboBox().setCurrentIndex(-1)
            self.videoPathLineEdit().clear()
            self.videoExtensionComboBox().setCurrentIndex(-1)
            self.imageFPSLineEdit().clear()
            self.progressBar().setValue(0)
