import cv2  # type: ignore
import pytest
from dipcoatimage.finitedepth import (
    SubstrateReference,
    HoughLinesParameters,
    RectSubstrate,
    get_samples_path,
)


@pytest.fixture
def rectsubst():
    ref_path = get_samples_path("ref1.png")
    img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
    tempROI = (200, 50, 1200, 200)
    substROI = (400, 175, 1000, 500)
    ref = SubstrateReference(img, tempROI, substROI)
    hparams = HoughLinesParameters(1, 0.01, 100)
    params = RectSubstrate.Parameters(hparams)
    subst = RectSubstrate(ref, parameters=params)
    return subst


def test_RectSubstrate_lines_notNone():
    ref_path = get_samples_path("ref1.png")
    img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
    tempROI = (200, 50, 1200, 200)
    substROI = (400, 175, 1000, 500)
    ref = SubstrateReference(img, tempROI, substROI)
    hparams = HoughLinesParameters(1, 0.01, 1000000000)
    params = RectSubstrate.Parameters(hparams)
    subst = RectSubstrate(ref, parameters=params)
    assert subst.lines().shape == (0, 1, 2)


def test_RectSubstrate_drawoptions(rectsubst):
    rectsubst.draw_options.draw_type = rectsubst.DrawMode.BINARY
    rectsubst.draw()

    rectsubst.draw_options.draw_type = rectsubst.DrawMode.EDGES
    rectsubst.draw()
