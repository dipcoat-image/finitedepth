import os

from dipcoatimage.finitedepth.util import get_data_path


def test_get_data_path():
    assert get_data_path("coat1.png").split(os.path.sep)[-4:] == [
        "dipcoatimage",
        "finitedepth",
        "data",
        "coat1.png",
    ]
