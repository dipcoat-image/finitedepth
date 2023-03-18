import cv2  # type: ignore
from dipcoatimage.finitedepth import (
    get_samples_path,
    SubstrateReference,
    Substrate,
    LayerArea,
)


def test_CoatingLayer_capbridge_broken():
    ref_img = cv2.imread(get_samples_path("ref3.png"))
    ref = SubstrateReference(ref_img, (100, 50, 1200, 200), (300, 100, 950, 600))
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
