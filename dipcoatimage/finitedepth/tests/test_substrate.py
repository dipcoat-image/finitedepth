import cv2  # type: ignore
from dataclasses import dataclass
from dipcoatimage.finitedepth import (
    SubstrateBase,
    get_samples_path,
    SubstrateReference,
    Substrate,
)
import pytest


def test_invalid_subclassing():
    class Valid(SubstrateBase):
        @dataclass(frozen=True)
        class Parameters:
            pass

        @dataclass
        class DrawOptions:
            pass

    with pytest.raises(TypeError):

        class NoParameters(SubstrateBase):
            @dataclass
            class DrawOptions:
                pass

    with pytest.raises(TypeError):

        class WrongParameters(SubstrateBase):
            Parameters = type

            @dataclass
            class DrawOptions:
                pass

    with pytest.raises(TypeError):

        class UnfrozenParameters(SubstrateBase):
            @dataclass
            class Parameters:
                pass

            @dataclass
            class DrawOptions:
                pass

    with pytest.raises(TypeError):

        class NoDrawOptions(SubstrateBase):
            @dataclass
            class Parameters:
                pass

    with pytest.raises(TypeError):

        class WrongDrawOptions(SubstrateBase):
            @dataclass(frozen=True)
            class Parameters:
                pass

            DrawOptions = type


def test_nestled_points():
    ref_path = get_samples_path("ref1.png")
    ref_img = cv2.imread(ref_path)

    subst1 = Substrate(SubstrateReference(ref_img, substrateROI=(400, 100, 1000, 500)))
    # test nestled_points
    assert subst1.nestled_points() == [(300, 0)]

    subst2 = Substrate(SubstrateReference(ref_img, substrateROI=(300, 50, 1100, 550)))
    # test nestled_points
    assert subst2.nestled_points() == [(400, 0)]
