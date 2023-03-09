from dipcoatimage.finitedepth import (
    get_samples_path,
    data_converter,
    ExperimentData,
)
import os
import yaml  # type: ignore


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
