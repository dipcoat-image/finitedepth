import os

from dipcoatimage.finitedepth import get_data_path


def test_get_data_path():
    fname = "coat1.png"
    dpath = get_data_path(fname)
    assert dpath.split(os.path.sep)[-4:] == [
        "dipcoatimage",
        "finitedepth",
        "data",
        fname,
    ]
    assert os.path.exists(dpath)
