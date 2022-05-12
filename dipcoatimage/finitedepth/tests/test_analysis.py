import csv
import os
import pytest
from dipcoatimage.finitedepth import get_samples_path
from dipcoatimage.finitedepth.analysis import ExperimentKind, experiment_kind, CSVWriter


def test_experiment_kind():
    img_path = [
        get_samples_path("coat1.png"),
    ]
    assert experiment_kind(img_path) == ExperimentKind.SingleImageExperiment

    imgs_path = [
        get_samples_path("coat1.png"),
        get_samples_path("coat2.png"),
    ]
    assert experiment_kind(imgs_path) == ExperimentKind.MultiImageExperiment

    vid_path = [
        get_samples_path("coat3.mp4"),
    ]
    assert experiment_kind(vid_path) == ExperimentKind.VideoExperiment

    empty_path = []
    assert experiment_kind(empty_path) == ExperimentKind.NullExperiment
    invalid_path = ["invalid.pdf"]
    assert experiment_kind(invalid_path) == ExperimentKind.NullExperiment
    vids_path = [
        get_samples_path("coat3.mp4"),
        get_samples_path("coat3.mp4"),
    ]
    assert experiment_kind(vids_path) == ExperimentKind.NullExperiment
    vidimg_path = [
        get_samples_path("coat3.mp4"),
        get_samples_path("coat1.png"),
    ]
    assert experiment_kind(vidimg_path) == ExperimentKind.NullExperiment


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
