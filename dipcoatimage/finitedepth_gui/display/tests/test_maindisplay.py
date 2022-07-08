from dipcoatimage.finitedepth_gui.display import MainDisplayWindow


def test_MainDisplayWindow_construction(qtbot):
    """Test that MainDisplayWindow can be constructed in test session"""
    # widget with internal thread must close the thread before exiting the test.
    w = MainDisplayWindow()
    w.close()
