import cv2  # type: ignore
import pytest
from dipcoatimage.finitedepth import (
    SubstrateReference,
    CannyParameters,
    HoughLinesParameters,
    RectSubstrate,
    get_samples_path,
)


@pytest.fixture
def rectsubst():
    ref_path = get_samples_path("ref1.png")
    img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
    tempROI = (200, 50, 1200, 200)
    substROI = (400, 100, 1000, 500)
    ref = SubstrateReference(img, tempROI, substROI)
    cparams = CannyParameters(50, 150)
    hparams = HoughLinesParameters(1, 0.01, 100)
    params = RectSubstrate.Parameters(cparams, hparams)
    subst = RectSubstrate(ref, parameters=params)
    return subst


def test_RectSubstrate(rectsubst):
    # test substrate analysis
    edge_lines = {
        k: (round(float(v[0]), 2), round(float(v[1]), 2))
        for k, v in rectsubst.edge_lines().items()
    }
    assert edge_lines == {
        rectsubst.Line_Left: (117.0, 0.0),
        rectsubst.Line_Right: (517.0, 0.0),
        rectsubst.Line_Top: (49.0, 1.57),
        rectsubst.Line_Bottom: (331.0, 1.57),
    }
    assert rectsubst.vertex_points() == {
        rectsubst.Point_TopLeft: (117, 48),
        rectsubst.Point_TopRight: (517, 48),
        rectsubst.Point_BottomLeft: (117, 330),
        rectsubst.Point_BottomRight: (517, 330),
    }


def test_RectSubstrate_lines_notNone():
    ref_path = get_samples_path("ref1.png")
    img = cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB)
    tempROI = (200, 50, 1200, 200)
    substROI = (400, 100, 1000, 500)
    ref = SubstrateReference(img, tempROI, substROI)
    cparams = CannyParameters(50, 150)
    hparams = HoughLinesParameters(1, 0.01, 1000000000)
    params = RectSubstrate.Parameters(cparams, hparams)
    subst = RectSubstrate(ref, parameters=params)
    assert subst.lines().shape == (0, 1, 2)


def test_RectSubstrate_drawoptions(rectsubst):
    rectsubst.draw_options.draw_type = rectsubst.Draw_Binary
    rectsubst.draw()

    rectsubst.draw_options.draw_type = rectsubst.Draw_Edges
    rectsubst.draw()
