import csv
import cv2  # type: ignore
import os
import pytest
from dipcoatimage.finitedepth import (
    ExperimentKind,
    experiment_kind,
    data_converter,
    ImportArgs,
    ReferenceArgs,
    SubstrateArgs,
    CoatingLayerArgs,
    ExperimentArgs,
    AnalysisArgs,
    ExperimentData,
    get_samples_path,
)
from dipcoatimage.finitedepth.util import dict_includes
from dipcoatimage.finitedepth.analysis import CSVWriter


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


def test_experiment_kind():
    img_path = [
        get_samples_path("coat1.png"),
    ]
    assert experiment_kind(img_path) == ExperimentKind.SINGLE_IMAGE

    imgs_path = [
        get_samples_path("coat1.png"),
        get_samples_path("coat2.png"),
    ]
    assert experiment_kind(imgs_path) == ExperimentKind.MULTI_IMAGE

    vid_path = [
        get_samples_path("coat3.mp4"),
    ]
    assert experiment_kind(vid_path) == ExperimentKind.VIDEO

    empty_path = []
    assert experiment_kind(empty_path) == ExperimentKind.NULL
    invalid_path = ["invalid.pdf"]
    assert experiment_kind(invalid_path) == ExperimentKind.NULL
    vids_path = [
        get_samples_path("coat3.mp4"),
        get_samples_path("coat3.mp4"),
    ]
    assert experiment_kind(vids_path) == ExperimentKind.NULL
    vidimg_path = [
        get_samples_path("coat3.mp4"),
        get_samples_path("coat1.png"),
    ]
    assert experiment_kind(vidimg_path) == ExperimentKind.NULL


def test_CSVWriter(tmp_path):
    datapath = os.path.join(tmp_path, "data.csv")
    headers = ["foo", "bar"]
    row1 = [1, 2]
    row2 = [3, 4]
    writer = CSVWriter(datapath, headers)
    writer.prepare()
    writer.write_data(row1)
    writer.write_data(row2)
    writer.terminate()

    assert os.path.exists(datapath)

    with open(datapath, "r") as datafile:
        reader = csv.reader(datafile)
        data_headers = next(reader)
        data_row1 = next(reader)
        data_row2 = next(reader)
        with pytest.raises(StopIteration):
            next(reader)

    assert data_headers == headers
    assert data_row1 == [str(i) for i in row1]
    assert data_row2 == [str(i) for i in row2]


def test_ReferenceArgs():
    refargs = ReferenceArgs(
        templateROI=(200, 100, 1200, 500),
        substrateROI=(300, 50, 1100, 600),
        draw_options=dict(substrateROI_thickness=2),
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
    refargs = ReferenceArgs(
        templateROI=(200, 100, 1200, 500),
        substrateROI=(300, 50, 1100, 600),
        draw_options=dict(substrateROI_thickness=2),
    )
    ref = refargs.as_reference(REF_IMG)

    substargs = SubstrateArgs(
        type=ImportArgs(name="RectSubstrate"),
        parameters=dict(
            HoughLines=dict(rho=1.0, theta=0.01, threshold=100),
        ),
        draw_options=dict(draw_lines=False),
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
    refargs = ReferenceArgs(
        templateROI=(200, 100, 1200, 500),
        substrateROI=(300, 50, 1100, 600),
        draw_options=dict(substrateROI_thickness=2),
    )
    ref = refargs.as_reference(REF_IMG)

    substargs = SubstrateArgs(
        type=ImportArgs(name="RectSubstrate"),
        parameters=dict(
            HoughLines=dict(rho=1.0, theta=0.01, threshold=100),
        ),
        draw_options=dict(draw_lines=False),
    )
    subst = substargs.as_substrate(ref)

    layerargs = CoatingLayerArgs(
        type=ImportArgs(name="RectLayerArea"),
        draw_options=dict(remove_substrate=True),
        deco_options=dict(paint_Left=False),
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
    refargs = ReferenceArgs(
        templateROI=(200, 100, 1200, 500),
        substrateROI=(300, 50, 1100, 600),
        draw_options=dict(substrateROI_thickness=2),
    )
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
    refargs = ReferenceArgs(
        templateROI=(200, 100, 1200, 500),
        substrateROI=(300, 50, 1100, 600),
        draw_options=dict(substrateROI_thickness=2),
    )
    substargs = SubstrateArgs()
    layerargs = CoatingLayerArgs()
    exptargs = ExperimentArgs()

    data_path = os.path.join(tmp_path, "expt_data1.csv")
    image_path = os.path.join(tmp_path, "expt_img1.png")
    video_path = os.path.join(tmp_path, "expt_img1.mp4")
    analargs = AnalysisArgs(
        data_path=data_path,
        image_path=image_path,
        video_path=video_path,
        fps=1,
    )
    data = ExperimentData(
        ref_path=get_samples_path("ref1.png"),
        coat_paths=[get_samples_path("coat1.png")],
        reference=refargs,
        substrate=substargs,
        coatinglayer=layerargs,
        experiment=exptargs,
        analysis=analargs,
    )
    data.analyze("test_ExperimentData_analyze_singleimage")

    assert os.path.exists(data_path)
    assert os.path.exists(image_path)
    assert os.path.exists(video_path)


def test_ExperimentData_analyze_multiimage(tmp_path):
    refargs = ReferenceArgs(
        templateROI=(200, 100, 1200, 500),
        substrateROI=(300, 50, 1100, 600),
        draw_options=dict(substrateROI_thickness=2),
    )
    substargs = SubstrateArgs()
    layerargs = CoatingLayerArgs()
    exptargs = ExperimentArgs()

    data_path = os.path.join(tmp_path, "expt_data1.csv")
    image_path = os.path.join(tmp_path, "expt_img1.png")
    video_path = os.path.join(tmp_path, "expt_img1.mp4")
    analargs = AnalysisArgs(
        data_path=data_path,
        image_path=image_path,
        video_path=video_path,
        fps=1,
    )
    data = ExperimentData(
        ref_path=get_samples_path("ref1.png"),
        coat_paths=[get_samples_path("coat1.png"), get_samples_path("coat1.png")],
        reference=refargs,
        substrate=substargs,
        coatinglayer=layerargs,
        experiment=exptargs,
        analysis=analargs,
    )
    data.analyze("test_ExperimentData_analyze_multiimage")

    assert os.path.exists(data_path)
    assert os.path.exists(image_path)
    assert os.path.exists(video_path)


def test_ExperimentData_analyze_video(tmp_path):
    refargs = ReferenceArgs(
        templateROI=(200, 100, 1200, 500),
        substrateROI=(300, 50, 1100, 600),
        draw_options=dict(substrateROI_thickness=2),
    )
    substargs = SubstrateArgs()
    layerargs = CoatingLayerArgs()
    exptargs = ExperimentArgs()

    data_path = os.path.join(tmp_path, "expt_data1.csv")
    image_path = os.path.join(tmp_path, "expt_img1.png")
    video_path = os.path.join(tmp_path, "expt_img1.mp4")
    analargs = AnalysisArgs(
        data_path=data_path,
        image_path=image_path,
        video_path=video_path,
        fps=1,
    )
    data = ExperimentData(
        ref_path=get_samples_path("ref3.png"),
        coat_paths=[get_samples_path("coat3.mp4")],
        reference=refargs,
        substrate=substargs,
        coatinglayer=layerargs,
        experiment=exptargs,
        analysis=analargs,
    )
    data.analyze("test_ExperimentData_analyze_video")

    assert os.path.exists(data_path)
    assert os.path.exists(image_path)
    assert os.path.exists(video_path)
