from dataclasses import dataclass
from dipcoatimage.finitedepth import CoatingLayerBase
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
