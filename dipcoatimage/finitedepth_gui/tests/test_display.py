import cv2
from PySide6.QtCore import Qt, QSize, QPoint
from dipcoatimage.finitedepth import get_samples_path
from dipcoatimage.finitedepth_gui.roimodel import ROIModel
from dipcoatimage.finitedepth_gui.display import (
    NDArrayROILabel,
    coords_label2pixmap,
    coords_pixmap2label,
)


def test_NDArrayROILabel_roi_conversion(qtbot):
    label = NDArrayROILabel()
    label.setAlignment(Qt.AlignCenter)
    img = cv2.imread(get_samples_path("ref1.png"), cv2.IMREAD_GRAYSCALE)
    label.resize(800, 600)
    label.setArray(img)

    assert label.size() == QSize(800, 600)
    assert label.pixmap().size() == QSize(750, 600)
    assert label._original_pixmap.size() == QSize(1407, 1125)
    originalROI = (100, 100, 200, 200)
    labelROI = tuple(map(int, label.originalROI2LabelROI(originalROI)))
    assert labelROI == (78, 53, 131, 106)
    assert tuple(map(int, label.labelROI2OriginalROI(labelROI))) == (99, 99, 198, 198)

    label.resize(1600, 1200)
    label.setArray(img)
    assert label.size() == QSize(1600, 1200)
    assert label.pixmap().size() == QSize(1407, 1125)
    assert label._original_pixmap.size() == QSize(1407, 1125)
    originalROI = (100, 100, 200, 200)
    labelROI = label.originalROI2LabelROI(originalROI)
    assert labelROI == (196.5, 137.5, 296.5, 237.5)
    assert tuple(map(int, label.labelROI2OriginalROI(labelROI))) == (100, 100, 200, 200)


def test_NDArrayROILabel_roiModels(qtbot):
    label = NDArrayROILabel()
    assert not label.roiModels()

    model1 = ROIModel()
    with qtbot.waitSignal(
        label.roiModelsChanged, check_params_cb=lambda l: l == [model1]
    ):
        label.addROIModel(model1)
    assert label.roiModels() == [model1]

    model2 = ROIModel()
    with qtbot.waitSignal(
        label.roiModelsChanged, check_params_cb=lambda l: l == [model2, model1]
    ):
        label.insertROIModel(0, model2)
    assert label.roiModels() == [model2, model1]

    with qtbot.waitSignal(
        label.roiModelsChanged, check_params_cb=lambda l: l == [model1]
    ):
        label.removeROIModel(model2)
    assert label.roiModels() == [model1]


def test_NDArrayROILabel_draw(qtbot):
    label = NDArrayROILabel()
    label.setAlignment(Qt.AlignCenter)
    label.resize(QSize(700, 500))

    model1 = ROIModel()
    label.addROIModel(model1)
    model2 = ROIModel()
    label.addROIModel(model2)

    img = cv2.imread(get_samples_path("coat1.png"), cv2.IMREAD_GRAYSCALE)
    label.setArray(img)

    with qtbot.waitSignals([model1.roiChanged, model2.roiChanged]):
        qtbot.mousePress(label, Qt.LeftButton, pos=QPoint(100, 100))
        qtbot.mouseRelease(label, Qt.LeftButton, pos=QPoint(200, 200))

    label.removeROIModel(model2)
    with qtbot.waitSignal(model1.roiChanged):
        qtbot.mousePress(label, Qt.LeftButton, pos=QPoint(100, 100))
        qtbot.mouseRelease(label, Qt.LeftButton, pos=QPoint(200, 200))
    with qtbot.assertNotEmitted(model2.roiChanged):
        qtbot.mousePress(label, Qt.LeftButton, pos=QPoint(100, 100))
        qtbot.mouseRelease(label, Qt.LeftButton, pos=QPoint(200, 200))

    # test that ROI is rsorted
    with qtbot.waitSignal(model1.roiChanged):
        qtbot.mousePress(label, Qt.LeftButton, pos=QPoint(200, 200))
        qtbot.mouseRelease(label, Qt.LeftButton, pos=QPoint(100, 100))


def test_NDArrayROILabel_update_nocrush_on_empty(qtbot):
    label = NDArrayROILabel()
    label.resize(800, 600)
    label.show()
    qtbot.waitExposed(label)
    label.update()


def test_NDArrayROILabel_update_nocrush_on_notempty(qtbot):
    label = NDArrayROILabel()
    img = cv2.imread(get_samples_path("ref1.png"), cv2.IMREAD_GRAYSCALE)
    label.setArray(img)
    label.show()
    qtbot.waitExposed(label)
    label.update()


def test_pixmap_label_coordinate_conversion():
    psize = QSize(600, 800)
    lsize = QSize(1203, 1003)
    alignment = Qt.AlignCenter

    pix_coord = (101, 103)
    lab_coord = coords_pixmap2label(pix_coord, psize, lsize, alignment)
    assert lab_coord == (402.5, 204.5)
    assert coords_label2pixmap(lab_coord, lsize, psize, alignment) == pix_coord
