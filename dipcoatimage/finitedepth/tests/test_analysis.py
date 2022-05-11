from dipcoatimage.finitedepth import get_samples_path
from dipcoatimage.finitedepth.analysis import ExperimentKind, experiment_kind


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
