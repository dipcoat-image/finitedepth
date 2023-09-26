import csv
import os
import pytest
from dipcoatimage.finitedepth import (
    ExperimentKind,
    experiment_kind,
    get_data_path,
)
from dipcoatimage.finitedepth.analysis import CSVWriter


def test_experiment_kind():
    img_path = [
        get_data_path("coat1.png"),
    ]
    assert experiment_kind(img_path) == ExperimentKind.SINGLE_IMAGE

    imgs_path = [
        get_data_path("coat1.png"),
        get_data_path("coat2.png"),
    ]
    assert experiment_kind(imgs_path) == ExperimentKind.MULTI_IMAGE

    vid_path = [
        get_data_path("coat3.mp4"),
    ]
    assert experiment_kind(vid_path) == ExperimentKind.VIDEO

    empty_path = []
    assert experiment_kind(empty_path) == ExperimentKind.NULL
    invalid_path = ["invalid.pdf"]
    assert experiment_kind(invalid_path) == ExperimentKind.NULL
    vids_path = [
        get_data_path("coat3.mp4"),
        get_data_path("coat3.mp4"),
    ]
    assert experiment_kind(vids_path) == ExperimentKind.NULL
    vidimg_path = [
        get_data_path("coat3.mp4"),
        get_data_path("coat1.png"),
    ]
    assert experiment_kind(vidimg_path) == ExperimentKind.NULL


def test_CSVWriter(tmp_path):
    datapath = os.path.join(tmp_path, "data.csv")
    headers = ["foo", "bar"]
    row1 = [1, 2]
    row2 = [3, 4]
    writer = CSVWriter(datapath, headers)
    next(writer)
    writer.send(row1)
    writer.send(row2)
    writer.close()

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
