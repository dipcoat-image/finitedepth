import os
from pytest import raises

from dipcoatimage.finitedepth.util import get_extended_line, get_samples_path


def test_get_extended_line():

    ret1 = get_extended_line((1080, 1920), (192, 108), (1920, 1080))
    assert ret1 == ((0, 0), (1920, 1080))
    ret2 = get_extended_line((1080, 1920), (300, 500), (400, 500))
    assert ret2 == ((0, 500), (1920, 500))
    ret3 = get_extended_line((1080, 1920), (900, 400), (900, 600))
    assert ret3 == ((900, 0), (900, 1080))

    with raises(ZeroDivisionError):
        get_extended_line((1080, 1920), (100, 100), (100, 100))


def test_get_samples_path():
    assert get_samples_path("coat1.png").split(os.path.sep)[-4:] == [
        "dipcoatimage",
        "finitedepth",
        "samples",
        "coat1.png",
    ]
