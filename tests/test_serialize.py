import os
from collections.abc import Iterable

import cv2  # type: ignore
import yaml  # type: ignore

from dipcoatimage.finitedepth import get_data_path
from dipcoatimage.finitedepth.serialize import (
    AnalysisArgs,
    CoatingLayerArgs,
    Config,
    ExperimentArgs,
    ImportArgs,
    ReferenceArgs,
    SubstrateArgs,
    data_converter,
)


def dict_includes(sup: dict, sub: dict):
    """Recursively check if *sup* is superset of *sub*."""
    for key, value in sub.items():
        if key not in sup:
            return False
        if isinstance(value, dict):
            if not dict_includes(sup[key], value):
                return False
        elif isinstance(value, Iterable):
            if not list(value) == list(sup[key]):
                return False
        else:
            if not value == sup[key]:
                return False
    return True


REFARGS = ReferenceArgs()
SUBSTARGS = SubstrateArgs()
LAYERARGS = CoatingLayerArgs()
EXPTARGS = ExperimentArgs()

REF_PATH = get_data_path("ref1.png")
REF_IMG = cv2.imread(REF_PATH, cv2.IMREAD_GRAYSCALE)
if REF_IMG is None:
    raise TypeError("Invalid reference image sample.")
_, REF_IMG = cv2.threshold(REF_IMG, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

COAT_PATH = get_data_path("coat1.png")
COAT_IMG = cv2.imread(COAT_PATH, cv2.IMREAD_GRAYSCALE)
if COAT_IMG is None:
    raise TypeError("Invalid coating layer image sample.")
_, COAT_IMG = cv2.threshold(COAT_IMG, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


def test_ReferenceArgs():
    refargs = ReferenceArgs(
        templateROI=(200, 50, 1200, 200),
        substrateROI=(400, 175, 1000, 500),
        draw_options=dict(substrateROI=dict(linewidth=2)),
    )
    ref = refargs.as_reference(REF_IMG)

    assert type(ref).__name__ == refargs.type.name
    assert ref.templateROI == refargs.templateROI
    assert ref.substrateROI == refargs.substrateROI
    assert dict_includes(
        data_converter.unstructure(ref.parameters),
        refargs.parameters,
    )
    assert dict_includes(
        data_converter.unstructure(ref.draw_options),
        refargs.draw_options,
    )


def test_SubstrateArgs():
    refargs = ReferenceArgs()
    ref = refargs.as_reference(REF_IMG)

    substargs = SubstrateArgs(
        type=ImportArgs(name="RectSubstrate"),
        parameters=dict(
            Sigma=3.0,
            Rho=1.0,
            Theta=0.01,
        ),
        draw_options=dict(sidelines=dict(linewidth=0)),
    )
    subst = substargs.as_substrate(ref)

    assert type(subst).__name__ == substargs.type.name
    assert dict_includes(
        data_converter.unstructure(subst.parameters),
        substargs.parameters,
    )
    assert dict_includes(
        data_converter.unstructure(subst.draw_options),
        substargs.draw_options,
    )


def test_CoatingLayerArgs():
    refargs = ReferenceArgs()
    ref = refargs.as_reference(REF_IMG)

    substargs = SubstrateArgs(
        type=ImportArgs(name="RectSubstrate"),
        parameters=dict(
            Sigma=3.0,
            Rho=1.0,
            Theta=0.01,
        ),
    )
    subst = substargs.as_substrate(ref)

    layerargs = CoatingLayerArgs(
        type=ImportArgs(name="RectLayerShape"),
        parameters=dict(
            KernelSize=(0, 0),
            ReconstructRadius=50,
            RoughnessMeasure="SDTW",
        ),
        draw_options=dict(paint="EMPTY"),
        deco_options=dict(layer=dict(linewidth=1)),
    )
    layer = layerargs.as_coatinglayer(COAT_IMG, subst)

    assert type(layer).__name__ == layerargs.type.name
    assert dict_includes(
        data_converter.unstructure(layer.parameters),
        layerargs.parameters,
    )
    assert dict_includes(
        data_converter.unstructure(layer.draw_options),
        layerargs.draw_options,
    )
    assert dict_includes(
        data_converter.unstructure(layer.deco_options),
        layerargs.deco_options,
    )


def test_ExperimentArgs():
    exptargs = ExperimentArgs()
    expt = exptargs.as_experiment()

    assert type(expt).__name__ == exptargs.type.name
    assert dict_includes(
        data_converter.unstructure(expt.parameters),
        exptargs.parameters,
    )


def test_Config_analyze_singleimage(tmp_path):
    analysisargs = AnalysisArgs(
        parameters=dict(
            ref_data=os.path.join(tmp_path, "ref.csv"),
            ref_visual=os.path.join(tmp_path, "ref.png"),
            subst_data=os.path.join(tmp_path, "subst.csv"),
            subst_visual=os.path.join(tmp_path, "subst.png"),
            layer_data=os.path.join(tmp_path, "layer.csv"),
            layer_visual=os.path.join(tmp_path, "layer.png"),
        )
    )
    data = Config(
        ref_path=get_data_path("ref1.png"),
        coat_path=get_data_path("coat1.png"),
        reference=REFARGS,
        substrate=SUBSTARGS,
        coatinglayer=LAYERARGS,
        experiment=EXPTARGS,
        analysis=analysisargs,
    )
    data.analyze("test_analyze_singleimage-to-singleimage")

    analysisargs = AnalysisArgs(
        parameters=dict(
            ref_data=os.path.join(tmp_path, "ref.csv"),
            ref_visual=os.path.join(tmp_path, "ref.png"),
            subst_data=os.path.join(tmp_path, "subst.csv"),
            subst_visual=os.path.join(tmp_path, "subst.png"),
            layer_data=os.path.join(tmp_path, "layer.csv"),
            layer_visual=os.path.join(tmp_path, "layer-%02d.png"),
        )
    )
    data = Config(
        ref_path=get_data_path("ref1.png"),
        coat_path=get_data_path("coat1.png"),
        reference=REFARGS,
        substrate=SUBSTARGS,
        coatinglayer=LAYERARGS,
        experiment=EXPTARGS,
        analysis=analysisargs,
    )
    data.analyze("test_analyze_singleimage-to-multiimage")

    analysisargs = AnalysisArgs(
        parameters=dict(
            ref_data=os.path.join(tmp_path, "ref.csv"),
            ref_visual=os.path.join(tmp_path, "ref.png"),
            subst_data=os.path.join(tmp_path, "subst.csv"),
            subst_visual=os.path.join(tmp_path, "subst.png"),
            layer_data=os.path.join(tmp_path, "layer.csv"),
            layer_visual=os.path.join(tmp_path, "layer.mp4"),
        ),
        fps=1.0,
    )
    data = Config(
        ref_path=get_data_path("ref1.png"),
        coat_path=get_data_path("coat1.png"),
        reference=REFARGS,
        substrate=SUBSTARGS,
        coatinglayer=LAYERARGS,
        experiment=EXPTARGS,
        analysis=analysisargs,
    )
    data.analyze("test_analyze_singleimage-to-video")


def test_Config_analyze_video(tmp_path):
    analysisargs = AnalysisArgs(
        parameters=dict(
            ref_data=os.path.join(tmp_path, "ref.csv"),
            ref_visual=os.path.join(tmp_path, "ref.png"),
            subst_data=os.path.join(tmp_path, "subst.csv"),
            subst_visual=os.path.join(tmp_path, "subst.png"),
            layer_data=os.path.join(tmp_path, "layer.csv"),
            layer_visual=os.path.join(tmp_path, "layer.mp4"),
        )
    )
    data = Config(
        ref_path=get_data_path("ref3.png"),
        coat_path=get_data_path("coat3.mp4"),
        reference=REFARGS,
        substrate=SUBSTARGS,
        coatinglayer=LAYERARGS,
        experiment=EXPTARGS,
        analysis=analysisargs,
    )
    data.analyze("test_analyze_video-to-video")


def test_Config_from_file():
    cwd = os.getcwd()

    try:
        os.chdir(get_data_path())
        with open("config.yml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in config.items():
            data_converter.structure(v, Config).analyze()
    finally:
        os.chdir(cwd)
