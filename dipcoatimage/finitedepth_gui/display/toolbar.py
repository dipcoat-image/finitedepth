import dipcoatimage.finitedepth_gui
from dipcoatimage.finitedepth_gui.core import VisualizationMode
import os
from PySide6.QtCore import Signal, QSize, Slot, QUrl
from PySide6.QtGui import QAction, QActionGroup, QIcon
from PySide6.QtWidgets import QToolBar, QComboBox, QLineEdit, QToolButton, QMenu, QStyle
from PySide6.QtMultimedia import (
    QCamera,
    QCameraDevice,
    QMediaDevices,
    QImageCapture,
    QMediaRecorder,
    QMediaFormat,
)
from typing import Optional


__all__ = [
    "DisplayWidgetToolBar",
    "get_icons_path",
]


class DisplayWidgetToolBar(QToolBar):
    """Toolbar to controll the overall display."""

    visualizationModeChanged = Signal(VisualizationMode)
    imageCaptured = Signal(str)
    videoRecorded = Signal(QUrl)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._visualizeAction = QAction("Toggle visualization")
        self._fastVisualizeAction = QAction("Toggle fast visualization")
        self._visualizeActionGroup = QActionGroup(self)
        self._camera = None
        self._cameraDeviceComboBox = QComboBox()
        self._cameraAction = QAction()
        self._imageCapture = None
        self._capturePathLineEdit = QLineEdit()
        self._captureFormatComboBox = QComboBox()
        self._captureButton = QToolButton()
        self._captureAndAddAction = QAction()
        self._mediaRecorder = None
        self._recordPathLineEdit = QLineEdit()
        self._recordFormatComboBox = QComboBox()
        self._recordButton = QToolButton()
        self._recordAndAddAction = QAction()

        self._visualizeAction.setCheckable(True)
        self._fastVisualizeAction.setCheckable(True)
        self._visualizeActionGroup.addAction(self._visualizeAction)
        self._visualizeActionGroup.addAction(self._fastVisualizeAction)
        self._visualizeActionGroup.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.ExclusiveOptional
        )
        self._visualizeActionGroup.triggered.connect(self._onVisualizeActionTrigger)
        self._cameraDeviceComboBox.currentIndexChanged.connect(
            self._onCameraDeviceComboBoxChange
        )
        self._cameraAction.setCheckable(True)
        self._cameraAction.toggled.connect(self._onCameraActionToggle)
        self._captureFormatComboBox.currentIndexChanged.connect(
            self._onCaptureFormatComboBoxChange
        )
        self._captureButton.clicked.connect(self._onCaptureButtonClick)
        self._recordPathLineEdit.editingFinished.connect(self._onRecordPathEdited)
        self._recordFormatComboBox.currentIndexChanged.connect(
            self._onRecordFormatComboBoxChange
        )
        self._recordButton.setCheckable(True)
        self._recordButton.toggled.connect(self._onRecordActionToggle)

        visIcon = QIcon()
        visIcon.addFile(get_icons_path("visualize.svg"), QSize(24, 24))
        self._visualizeAction.setIcon(visIcon)

        fastVisIcon = QIcon()
        fastVisIcon.addFile(get_icons_path("fastvisualize.svg"), QSize(24, 24))
        self._fastVisualizeAction.setIcon(fastVisIcon)

        self._cameraDeviceComboBox.setPlaceholderText("Select camera")
        self._cameraAction.setToolTip("Toggle Camera")
        cameraActionIcon = QIcon()
        cameraActionIcon.addFile(get_icons_path("camera.svg"), QSize(24, 24))
        self._cameraAction.setIcon(cameraActionIcon)

        self._capturePathLineEdit.setPlaceholderText("Image capture path")
        self._captureFormatComboBox.setPlaceholderText("Image format")
        self._captureButton.setToolTip("Capture image")
        captureActionIcon = QIcon()
        captureActionIcon.addFile(get_icons_path("capture.svg"), QSize(24, 24))
        self._captureButton.setIcon(captureActionIcon)
        self._captureButton.setMenu(QMenu(self))
        self._captureAndAddAction.setText("Capture and add to data")
        self._captureAndAddAction.setCheckable(True)
        self._captureButton.menu().addAction(self._captureAndAddAction)
        self._captureButton.setPopupMode(QToolButton.MenuButtonPopup)

        self._recordPathLineEdit.setPlaceholderText("Video record path")
        self._recordFormatComboBox.setPlaceholderText("Video format")
        self._recordButton.setToolTip("Record video")
        self._recordButton.setCheckable(True)
        recordActionIcon = QIcon()
        recordActionIcon.addFile(get_icons_path("record.svg"), QSize(24, 24))
        self._recordButton.setIcon(recordActionIcon)
        self._recordButton.setMenu(QMenu(self))
        self._recordAndAddAction.setText("Record and add to data")
        self._recordAndAddAction.setCheckable(True)
        self._recordButton.menu().addAction(self._recordAndAddAction)
        self._recordButton.setPopupMode(QToolButton.MenuButtonPopup)

        self.addAction(self._visualizeAction)
        self.addAction(self._fastVisualizeAction)
        self.addSeparator()
        self.addSeparator()
        self.addWidget(self._cameraDeviceComboBox)
        self.addAction(self._cameraAction)
        self.addSeparator()
        self.addWidget(self._capturePathLineEdit)
        self.addWidget(self._captureFormatComboBox)
        self.addWidget(self._captureButton)
        self.addSeparator()
        self.addWidget(self._recordPathLineEdit)
        self.addWidget(self._recordFormatComboBox)
        self.addWidget(self._recordButton)

        self.loadCameraDevices()
        self.loadImageFileFormats()
        self.loadMediaFileFormats()

    def _onVisualizeActionTrigger(self, action: QAction):
        if action.isChecked() and action == self._visualizeAction:
            mode = VisualizationMode.FULL
        elif action.isChecked() and action == self._fastVisualizeAction:
            mode = VisualizationMode.FAST
        else:
            mode = VisualizationMode.OFF
        self.visualizationModeChanged.emit(mode)

    @Slot(VisualizationMode)
    def setVisualizationMode(self, mode: VisualizationMode):
        if mode == VisualizationMode.OFF:
            self._visualizeAction.setChecked(False)
            self._fastVisualizeAction.setChecked(False)
        elif mode == VisualizationMode.FAST:
            self._visualizeAction.setChecked(False)
            self._fastVisualizeAction.setChecked(True)
        elif mode == VisualizationMode.FULL:
            self._fastVisualizeAction.setChecked(False)
            self._visualizeAction.setChecked(True)

    def camera(self) -> Optional[QCamera]:
        return self._camera

    def setCamera(self, camera: Optional[QCamera]):
        oldCamera = self.camera()
        if oldCamera is not None:
            oldCamera.cameraDeviceChanged.disconnect(  # type: ignore[attr-defined]
                self._onCameraDeviceChange
            )
            oldCamera.activeChanged.disconnect(  # type: ignore[attr-defined]
                self._onCameraActiveChange
            )
        self._camera = camera
        if camera is not None:
            camera.cameraDeviceChanged.connect(  # type: ignore[attr-defined]
                self._onCameraDeviceChange
            )
            camera.activeChanged.connect(  # type: ignore[attr-defined]
                self._onCameraActiveChange
            )

    def loadCameraDevices(self):
        self._cameraDeviceComboBox.clear()
        for device in QMediaDevices.videoInputs():
            name = device.description()
            self._cameraDeviceComboBox.addItem(name, userData=device)

    def _onCameraDeviceComboBoxChange(self, index: int):
        device = self._cameraDeviceComboBox.itemData(index)
        if not isinstance(device, QCameraDevice):
            device = QCameraDevice()
        camera = self.camera()
        if camera is not None:
            camera.setCameraDevice(device)

    @Slot(QCameraDevice)
    def _onCameraDeviceChange(self, device: QCameraDevice):
        index = self._cameraDeviceComboBox.findData(device)
        self._cameraDeviceComboBox.setCurrentIndex(index)

    def _onCameraActionToggle(self, checked: bool):
        camera = self.camera()
        if camera is not None:
            camera.setActive(checked)

    @Slot(bool)
    def _onCameraActiveChange(self, active: bool):
        self._cameraAction.setChecked(active)

    def imageCapture(self) -> Optional[QImageCapture]:
        return self._imageCapture

    def setImageCapture(self, imageCapture: Optional[QImageCapture]):
        oldImageCapture = self.imageCapture()
        if oldImageCapture is not None:
            oldImageCapture.fileFormatChanged.disconnect(  # type: ignore[attr-defined]
                self._onCaptureFileFormatChange
            )
            oldImageCapture.imageSaved.disconnect(  # type: ignore[attr-defined]
                self._onImageSave
            )
        self._imageCapture = imageCapture
        if imageCapture is not None:
            imageCapture.fileFormatChanged.connect(  # type: ignore[attr-defined]
                self._onCaptureFileFormatChange
            )
            imageCapture.imageSaved.connect(  # type: ignore[attr-defined]
                self._onImageSave
            )

    def loadImageFileFormats(self):
        self._captureFormatComboBox.clear()
        for form in QImageCapture.supportedFormats():
            name = QImageCapture.fileFormatName(form)
            self._captureFormatComboBox.addItem(name, userData=form)

    def _onCaptureFormatComboBoxChange(self, index: int):
        form = self._captureFormatComboBox.itemData(index)
        if not isinstance(form, QImageCapture.FileFormat):
            form = QImageCapture.FileFormat(0)
        imageCapture = self.imageCapture()
        if imageCapture is not None:
            imageCapture.setFileFormat(form)

    @Slot(QImageCapture.FileFormat)
    def _onCaptureFileFormatChange(self):
        imageCapture = self.imageCapture()
        if imageCapture is not None:
            form = imageCapture.fileFormat()
        index = self._captureFormatComboBox.findData(form)
        self._captureFormatComboBox.setCurrentIndex(index)

    def _onCaptureButtonClick(self):
        path = self._capturePathLineEdit.text()
        imageCapture = self.imageCapture()
        if imageCapture is not None:
            imageCapture.captureToFile(os.path.abspath(path))

    @Slot(int, str)
    def _onImageSave(self, id: int, path: str):
        if id != -1 and self._captureAndAddAction.isChecked():
            self.imageCaptured.emit(path)

    def mediaRecorder(self) -> Optional[QMediaRecorder]:
        return self._mediaRecorder

    def setMediaRecorder(self, recorder: Optional[QMediaRecorder]):
        oldRecorder = self.mediaRecorder()
        if oldRecorder is not None:
            oldRecorder.actualLocationChanged.disconnect(  # type: ignore[attr-defined]
                self._onRecordLocationChange
            )
            oldRecorder.mediaFormatChanged.disconnect(  # type: ignore[attr-defined]
                self._onMediaFormatChange
            )
            oldRecorder.recorderStateChanged.disconnect(  # type: ignore[attr-defined]
                self._onRecorderStateChange
            )
        self._mediaRecorder = recorder
        if recorder is not None:
            recorder.actualLocationChanged.connect(  # type: ignore[attr-defined]
                self._onRecordLocationChange
            )
            recorder.mediaFormatChanged.connect(  # type: ignore[attr-defined]
                self._onMediaFormatChange
            )
            recorder.recorderStateChanged.connect(  # type: ignore[attr-defined]
                self._onRecorderStateChange
            )

    def loadMediaFileFormats(self):
        self._recordFormatComboBox.clear()
        for form in QMediaFormat().supportedFileFormats(QMediaFormat.Encode):
            name = QMediaFormat.fileFormatName(form)
            self._recordFormatComboBox.addItem(name, userData=form)

    def _onRecordPathEdited(self):
        path = os.path.abspath(self._recordPathLineEdit.text())
        self.mediaRecorder().setOutputLocation(QUrl.fromLocalFile(path))

    @Slot(QUrl)
    def _onRecordLocationChange(self, location: QUrl):
        path = os.path.relpath(location.toLocalFile())
        self._recordPathLineEdit.setText(path)

    def _onRecordFormatComboBoxChange(self, index: int):
        form = self._recordFormatComboBox.itemData(index)
        if not isinstance(form, QMediaFormat.FileFormat):
            form = QMediaFormat.FileFormat(0)
        mediaRecorder = self.mediaRecorder()
        if mediaRecorder is not None:
            mediaRecorder.mediaFormat().setFileFormat(form)

    @Slot()
    def _onMediaFormatChange(self):
        recorder = self.mediaRecorder()
        if recorder is not None:
            form = recorder.mediaFormat().fileFormat()
            index = self._recordFormatComboBox.findData(form)
            self._recordFormatComboBox.setCurrentIndex(index)

    def _onRecordActionToggle(self, checked: bool):
        recorder = self.mediaRecorder()
        if recorder is not None:
            if checked:
                recorder.record()
            else:
                recorder.stop()

    @Slot(QMediaRecorder.RecorderState)
    def _onRecorderStateChange(self, state: QMediaRecorder.RecorderState):
        if state == QMediaRecorder.RecorderState.RecordingState:
            self._recordButton.setChecked(True)
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop)
        elif state == QMediaRecorder.RecorderState.StoppedState:
            recorder = self.mediaRecorder()
            if recorder is not None and self._recordAndAddAction.isChecked():
                self.videoRecorded.emit(recorder.actualLocation())
            self._recordButton.setChecked(False)
            icon = QIcon()
            icon.addFile(get_icons_path("record.svg"), QSize(24, 24))
        else:
            raise TypeError(f"Invalid recorder state: {state}")
        self._recordButton.setIcon(icon)


def get_icons_path(*paths: str) -> str:
    """
    Get the absolute path to the directory where the icon files are
    stored.

    Parameters
    ==========

    paths
        Subpaths under ``dipcoatimage/finitedepth_gui/icons/`` directory.

    Returns
    =======

    path
        Absolute path to the icon depending on the user's system.

    """
    module_path = os.path.abspath(dipcoatimage.finitedepth_gui.__file__)
    module_path = os.path.split(module_path)[0]
    sample_dir = os.path.join(module_path, "icons")
    sample_dir = os.path.normpath(sample_dir)
    sample_dir = os.path.normcase(sample_dir)

    path = os.path.join(sample_dir, *paths)
    return path
