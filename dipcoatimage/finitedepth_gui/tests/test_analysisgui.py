from dipcoatimage.finitedepth import get_samples_path
from dipcoatimage.finitedepth_gui import AnalysisGUI
import pytest


@pytest.mark.skip(reason="PySide6 6.4 breaks this test")
def test_issue65(qtbot):
    """Test that changing options does not stop video."""
    window = AnalysisGUI()

    player = window.mainDisplayWindow().videoPlayer()

    window.masterControlWidget().referenceWidget().pathLineEdit().setText(
        get_samples_path("ref3.png")
    )
    window.masterControlWidget().referenceWidget().pathLineEdit().editingFinished.emit()
    window.masterControlWidget().experimentWidget().addCoatPath(
        get_samples_path("coat3.mp4")
    )

    player.setLoops(player.Infinite)
    with qtbot.waitSignal(
        player.playbackStateChanged,
        check_params_cb=lambda state: state == player.PlayingState,
    ):
        player.play()

    window.mainDisplayWindow().displayToolBar().visualizeAction().trigger()
    window.close()

    assert player.playbackState() == player.PlayingState
