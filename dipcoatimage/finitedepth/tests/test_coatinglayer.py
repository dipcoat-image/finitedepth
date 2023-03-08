import cv2  # type: ignore
import numpy as np
from dipcoatimage.finitedepth import (
    get_samples_path,
    SubstrateReference,
    Substrate,
    LayerArea,
)


def test_CoatingLayer_template_matching():
    ref_img = cv2.imread(get_samples_path("ref1.png"))
    ref = SubstrateReference(ref_img, (200, 50, 1200, 200), (400, 100, 1000, 500))
    subst = Substrate(ref)
    coat_img = cv2.imread(get_samples_path("coat1.png"))
    coat = LayerArea(coat_img, subst)

    score, point = coat.match_substrate()
    assert score == 0.0
    assert np.all(point == np.array([200, 501]))
    assert coat.substrate_point() == (400, 551)


def test_CoatingLayer_capbridge_broken():
    ref_img = cv2.imread(get_samples_path("ref3.png"))
    ref = SubstrateReference(ref_img, (100, 50, 1200, 600), (200, 50, 1100, 800))
    subst = Substrate(ref)

    vid_path = get_samples_path("coat3.mp4")
    cap = cv2.VideoCapture(vid_path)
    _, unbroke_img = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    _, broke_img = cap.read()
    cap.release()

    unbroke_layer = LayerArea(unbroke_img, subst)
    assert not unbroke_layer.capbridge_broken()
    broke_layer = LayerArea(broke_img, subst)
    assert broke_layer.capbridge_broken()
