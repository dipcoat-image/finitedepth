import os

from dipcoatimage.finitedepth.util import get_samples_path


def test_get_samples_path():
    assert get_samples_path("coat1.png").split(os.path.sep)[-4:] == [
        "dipcoatimage",
        "finitedepth",
        "data",
        "coat1.png",
    ]
