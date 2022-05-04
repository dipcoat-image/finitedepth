from dataclasses import dataclass
from dipcoatimage.finitedepth import SubstrateReferenceBase
import pytest


def test_invalid_subclassing():
    class Valid(SubstrateReferenceBase):
        @dataclass(frozen=True)
        class Parameters:
            pass

        @dataclass
        class DrawOptions:
            pass

    with pytest.raises(TypeError):

        class NoParameters(SubstrateReferenceBase):
            @dataclass
            class DrawOptions:
                pass

    with pytest.raises(TypeError):

        class WrongParameters(SubstrateReferenceBase):
            Parameters = type

            @dataclass
            class DrawOptions:
                pass

    with pytest.raises(TypeError):

        class UnfrozenParameters(SubstrateReferenceBase):
            @dataclass
            class Parameters:
                pass

            @dataclass
            class DrawOptions:
                pass

    with pytest.raises(TypeError):

        class NoDrawOptions(SubstrateReferenceBase):
            @dataclass
            class Parameters:
                pass

    with pytest.raises(TypeError):

        class WrongDrawOptions(SubstrateReferenceBase):
            @dataclass(frozen=True)
            class Parameters:
                pass

            DrawOptions = type
