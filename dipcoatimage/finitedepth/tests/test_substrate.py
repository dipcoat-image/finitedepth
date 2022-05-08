import cv2  # type: ignore
from dipcoatimage.finitedepth import (
    get_samples_path,
    SubstrateReference,
    Substrate,
)


def test_nestled_points():
    ref_path = get_samples_path("ref1.png")
    ref_img = cv2.imread(ref_path)

    subst1 = Substrate(SubstrateReference(ref_img, substrateROI=(400, 100, 1000, 500)))
    # test nestled_points
    assert subst1.nestled_points() == [(300, 0)]

    subst2 = Substrate(SubstrateReference(ref_img, substrateROI=(300, 50, 1100, 550)))
    # test nestled_points
    assert subst2.nestled_points() == [(400, 0)]
