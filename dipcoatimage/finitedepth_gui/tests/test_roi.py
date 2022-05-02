from dipcoatimage.finitedepth_gui import ROIModel, ROIWidget


def test_ROIModel_signals(qtbot):
    model = ROIModel()
    assert model.roi() == (0, 0, None, None)

    newroi = (1, 2, 3, 4)
    with qtbot.waitSignal(
        model.roiChanged,
        check_params_cb=lambda x1, y1, x2, y2: (x1, y1, x2, y2) == newroi,
    ):
        model.setROI(*newroi)
    assert model.roi() == newroi

    newroi = (3, 4, None, None)
    with qtbot.waitSignal(
        model.roiChanged,
        check_params_cb=lambda x1, y1, x2, y2: (x1, y1, x2, y2) == newroi,
    ):
        model.setROI(*newroi)
    assert model.roi() == newroi


def test_ROIWidget_control(qtbot):
    widget = ROIWidget()
    model = widget.roiModel()
    widget.setROIMaximum(10, 10)

    newroi = (1, 2, 3, 4)
    with qtbot.waitSignal(
        model.roiChanged,
        check_params_cb=lambda x1, y1, x2, y2: (x1, y1, x2, y2) == newroi,
    ):
        widget.setROI(*newroi)
    assert model.roi() == newroi

    x1 = 5
    newroi = (x1, 2, 3, 4)
    with qtbot.waitSignal(
        model.roiChanged,
        check_params_cb=lambda x1, y1, x2, y2: (x1, y1, x2, y2) == newroi,
    ):
        widget.x1SpinBox().setValue(x1)
    assert model.roi() == newroi

    y1 = 6
    newroi = (x1, y1, 3, 4)
    with qtbot.waitSignal(
        model.roiChanged,
        check_params_cb=lambda x1, y1, x2, y2: (x1, y1, x2, y2) == newroi,
    ):
        widget.y1SpinBox().setValue(y1)
    assert model.roi() == newroi

    x2 = 5
    newroi = (x1, y1, x2, 4)
    with qtbot.waitSignal(
        model.roiChanged,
        check_params_cb=lambda x1, y1, x2, y2: (x1, y1, x2, y2) == newroi,
    ):
        widget.x2SpinBox().setValue(x2)
    assert model.roi() == newroi

    y2 = 5
    newroi = (x1, y1, x2, y2)
    with qtbot.waitSignal(
        model.roiChanged,
        check_params_cb=lambda x1, y1, x2, y2: (x1, y1, x2, y2) == newroi,
    ):
        widget.y2SpinBox().setValue(y2)
    assert model.roi() == newroi


def test_ROIWidget_view(qtbot):
    widget = ROIWidget()
    model = widget.roiModel()

    maxroi = (10, 20)
    newroi = (0, 0, maxroi[0], maxroi[1])
    with qtbot.waitSignals(
        [widget.roiMaximumChanged, widget.roiChanged],
        check_params_cbs=[
            lambda w, h: (w, h) == maxroi,
            lambda x1, y1, x2, y2: (x1, y1, x2, y2) == newroi,
        ],
    ):
        widget.setROIMaximum(*maxroi)
    assert widget.x1SpinBox().maximum() == maxroi[0]
    assert widget.y1SpinBox().maximum() == maxroi[1]
    assert widget.x2SpinBox().maximum() == maxroi[0]
    assert widget.y2SpinBox().maximum() == maxroi[1]
    assert widget.displayedROI() == newroi

    newroi = (1, 2, 3, 4)
    with qtbot.waitSignal(
        widget.roiChanged,
        check_params_cb=lambda x1, y1, x2, y2: (x1, y1, x2, y2) == newroi,
    ):
        model.setROI(*newroi)
    assert widget.displayedROI() == newroi

    newroi = (5, 6, None, None)
    resultroi = (newroi[0], newroi[1], maxroi[0], maxroi[1])
    with qtbot.waitSignal(
        widget.roiChanged,
        check_params_cb=lambda x1, y1, x2, y2: (x1, y1, x2, y2) == resultroi,
    ):
        model.setROI(*newroi)
    assert widget.displayedROI() == resultroi
