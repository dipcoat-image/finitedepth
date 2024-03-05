import os
import subprocess

import yaml

from finitedepth import get_sample_path


def test_coatingimage(tmp_path):
    imgpath = tmp_path / "layer.png"
    datapath = tmp_path / "layer.csv"
    config = dict(
        data1=dict(
            type="CoatingImage",
            referencePath=get_sample_path("ref.png"),
            targetPath=get_sample_path("coat.png"),
            output=dict(layerImage=str(imgpath), layerData=str(datapath)),
        )
    )
    path = tmp_path / "config.yml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    code = subprocess.call(
        [
            "finitedepth",
            "analyze",
            path,
        ],
    )
    assert not code
    assert os.path.exists(imgpath)
    assert os.path.exists(datapath)


def test_coatingvideo(tmp_path):
    vidpath = tmp_path / "layer.mp4"
    datapath = tmp_path / "layer.csv"
    config = dict(
        data1=dict(
            type="CoatingVideo",
            referencePath=get_sample_path("ref.png"),
            targetPath=get_sample_path("coat.mp4"),
            output=dict(layerVideo=str(vidpath), layerData=str(datapath)),
        )
    )
    path = tmp_path / "config.yml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    code = subprocess.call(
        [
            "finitedepth",
            "analyze",
            path,
        ],
    )
    assert not code
    assert os.path.exists(vidpath)
    assert os.path.exists(datapath)
