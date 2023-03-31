from collections.abc import Iterable
import cv2  # type: ignore
from dipcoatimage.finitedepth import (
    get_samples_path,
    data_converter,
    ImportArgs,
    ReferenceArgs,
    SubstrateArgs,
    CoatingLayerArgs,
    ExperimentArgs,
    AnalysisArgs,
    ExperimentData,
)
import os
import yaml  # type: ignore


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

REF_PATH = get_samples_path("ref1.png")
REF_IMG = cv2.imread(REF_PATH)
if REF_IMG is None:
    raise TypeError("Invalid reference image sample.")
REF_IMG = cv2.cvtColor(REF_IMG, cv2.COLOR_BGR2RGB)

COAT_PATH = get_samples_path("coat1.png")
COAT_IMG = cv2.imread(COAT_PATH)
if COAT_IMG is None:
    raise TypeError("Invalid coating layer image sample.")
COAT_IMG = cv2.cvtColor(COAT_IMG, cv2.COLOR_BGR2RGB)


def test_ReferenceArgs():
    refargs = ReferenceArgs(
        templateROI=(200, 50, 1200, 200),
        substrateROI=(400, 175, 1000, 500),
        draw_options=dict(substrateROI=dict(thickness=2)),
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
            HoughLines=dict(rho=1.0, theta=0.01, threshold=100),
        ),
        draw_options=dict(sides=dict(thickness=0)),
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
            HoughLines=dict(rho=1.0, theta=0.01, threshold=100),
        ),
    )
    subst = substargs.as_substrate(ref)

    layerargs = CoatingLayerArgs(
        type=ImportArgs(name="RectLayerShape"),
        parameters=dict(
            MorphologyClosing=dict(kernelSize=(0, 0)),
            ReconstructRadius=50,
        ),
        draw_options=dict(background="BINARY"),
        deco_options=dict(layer=dict(thickness=1)),
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
    refargs = ReferenceArgs()
    ref = refargs.as_reference(REF_IMG)

    substargs = SubstrateArgs()
    subst = substargs.as_substrate(ref)

    layerargs = CoatingLayerArgs()
    exptargs = ExperimentArgs()
    expt = exptargs.as_experiment(subst, *layerargs.as_structured_args())

    assert type(expt).__name__ == exptargs.type.name
    assert dict_includes(
        data_converter.unstructure(expt.parameters),
        exptargs.parameters,
    )


def test_ExperimentData_analyze_singleimage(tmp_path):
    data_path = os.path.join(tmp_path, "data.csv")
    image_path = os.path.join(tmp_path, "img.png")
    video_path = os.path.join(tmp_path, "vid.mp4")
    analargs = AnalysisArgs(
        data_path=data_path,
        image_path=image_path,
        video_path=video_path,
        fps=1,
    )
    data = ExperimentData(
        ref_path=get_samples_path("ref1.png"),
        coat_paths=[get_samples_path("coat1.png")],
        reference=REFARGS,
        substrate=SUBSTARGS,
        coatinglayer=LAYERARGS,
        experiment=EXPTARGS,
        analysis=analargs,
    )
    data.analyze("test_ExperimentData_analyze_singleimage")

    assert os.path.exists(data_path)
    assert os.path.exists(image_path)
    assert os.path.exists(video_path)


def test_ExperimentData_analyze_multiimage(tmp_path):
    data_path = os.path.join(tmp_path, "data.csv")
    image_path = os.path.join(tmp_path, "img.png")
    video_path = os.path.join(tmp_path, "vid.mp4")
    analargs = AnalysisArgs(
        data_path=data_path,
        image_path=image_path,
        video_path=video_path,
        fps=1,
    )
    data = ExperimentData(
        ref_path=get_samples_path("ref1.png"),
        coat_paths=[get_samples_path("coat1.png"), get_samples_path("coat1.png")],
        reference=REFARGS,
        substrate=SUBSTARGS,
        coatinglayer=LAYERARGS,
        experiment=EXPTARGS,
        analysis=analargs,
    )
    data.analyze("test_ExperimentData_analyze_multiimage")

    assert os.path.exists(data_path)
    assert os.path.exists(image_path)
    assert os.path.exists(video_path)


def test_ExperimentData_analyze_video(tmp_path):
    data_path = os.path.join(tmp_path, "data.csv")
    image_path = os.path.join(tmp_path, "img.png")
    video_path = os.path.join(tmp_path, "vid.mp4")
    analargs = AnalysisArgs(
        data_path=data_path,
        image_path=image_path,
        video_path=video_path,
        fps=1,
    )
    data = ExperimentData(
        ref_path=get_samples_path("ref3.png"),
        coat_paths=[get_samples_path("coat3.mp4")],
        reference=REFARGS,
        substrate=SUBSTARGS,
        coatinglayer=LAYERARGS,
        experiment=EXPTARGS,
        analysis=analargs,
    )
    data.analyze("test_ExperimentData_analyze_video")

    assert os.path.exists(data_path)
    assert os.path.exists(image_path)
    assert os.path.exists(video_path)


def test_ExperimentData_from_file():
    cwd = os.getcwd()

    try:
        os.chdir(get_samples_path())
        with open("config.yml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in config.items():
            data_converter.structure(v, ExperimentData).analyze()
    finally:
        os.chdir(cwd)
