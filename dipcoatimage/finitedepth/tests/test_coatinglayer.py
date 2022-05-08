import cv2  # type: ignore
from dataclasses import dataclass
from dipcoatimage.finitedepth import (
    get_samples_path,
    CoatingLayerBase,
    SubstrateReference,
    Substrate,
    LayerArea,
)
import pytest


def test_invalid_subclassing():
    class Valid(CoatingLayerBase):
        @dataclass(frozen=True)
        class Parameters:
            pass

        @dataclass
        class DrawOptions:
            pass

        @dataclass
        class DecoOptions:
            pass

        @dataclass
        class Data:
            pass

    with pytest.raises(TypeError):

        class NoParameters(CoatingLayerBase):
            @dataclass
            class DrawOptions:
                pass

            @dataclass
            class DecoOptions:
                pass

            @dataclass
            class Data:
                pass

    with pytest.raises(TypeError):

        class WrongParameters(CoatingLayerBase):
            Parameters = type

            @dataclass
            class DrawOptions:
                pass

            @dataclass
            class DecoOptions:
                pass

            @dataclass
            class Data:
                pass

    with pytest.raises(TypeError):

        class UnfrozenParameters(CoatingLayerBase):
            @dataclass
            class Parameters:
                pass

            @dataclass
            class DrawOptions:
                pass

            @dataclass
            class DecoOptions:
                pass

            @dataclass
            class Data:
                pass

    with pytest.raises(TypeError):

        class NoDrawOptions(CoatingLayerBase):
            @dataclass
            class Parameters:
                pass

            @dataclass
            class DecoOptions:
                pass

            @dataclass
            class Data:
                pass

    with pytest.raises(TypeError):

        class WrongDrawOptions(CoatingLayerBase):
            @dataclass(frozen=True)
            class Parameters:
                pass

            DrawOptions = type

            @dataclass
            class DecoOptions:
                pass

            @dataclass
            class Data:
                pass

    with pytest.raises(TypeError):

        class NoDecoOptions(CoatingLayerBase):
            @dataclass
            class Parameters:
                pass

            @dataclass
            class DrawOptions:
                pass

            @dataclass
            class Data:
                pass

    with pytest.raises(TypeError):

        class WrongDecoOptions(CoatingLayerBase):
            @dataclass(frozen=True)
            class Parameters:
                pass

            @dataclass
            class DrawOptions:
                pass

            DecoOptions = type

            @dataclass
            class Data:
                pass

    with pytest.raises(TypeError):

        class NoData(CoatingLayerBase):
            @dataclass
            class Parameters:
                pass

            @dataclass
            class DrawOptions:
                pass

            @dataclass
            class DecoOptions:
                pass

    with pytest.raises(TypeError):

        class WrongData(CoatingLayerBase):
            @dataclass
            class Parameters:
                pass

            @dataclass
            class DrawOptions:
                pass

            @dataclass
            class DecoOptions:
                pass

            Data = type


def test_CoatingLayer_template_matching():
    ref_img = cv2.imread(get_samples_path("ref1.png"))
    ref = SubstrateReference(ref_img, (200, 50, 1200, 200), (400, 100, 1000, 500))
    subst = Substrate(ref)
    coat_img = cv2.imread(get_samples_path("coat1.png"))
    coat = LayerArea(coat_img, subst)

    assert coat.template_point() == (200, 501)
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
