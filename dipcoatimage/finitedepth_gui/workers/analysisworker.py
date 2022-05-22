import cv2  # type: ignore
from dipcoatimage.finitedepth import ExperimentBase
import dataclasses
from dipcoatimage.finitedepth.analysis import ExperimentKind, AnalysisArgs
import os
from PySide6.QtCore import Slot, Signal
from typing import Optional, List
from .base import WorkerBase


__all__ = [
    "AnalysisWorker",
]


class AnalysisWorker(WorkerBase):
    """
    Worker to analyze the coated substrate files.

    Data for analysis are:

    1. :meth:`experiment`
    2. :meth:`paths`
    3. :meth:`analysisArgs`

    """

    progressMaximumChanged = Signal(int)
    progressValueChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._experiment = None
        self._paths = []
        self._expt_kind = ExperimentKind.NullExperiment
        self._analysisArgs = AnalysisArgs()

    def experiment(self) -> Optional[ExperimentBase]:
        return self._experiment

    def paths(self) -> List[str]:
        return self._paths

    def experimentKind(self) -> ExperimentKind:
        return self._expt_kind

    def analysisArgs(self) -> AnalysisArgs:
        return self._analysisArgs

    @Slot(object)
    def setExperiment(self, expt: Optional[ExperimentBase]):
        self._experiment = expt

    def setPaths(self, paths: List[str], kind: ExperimentKind):
        self._paths = paths
        self._expt_kind = kind

    def setAnalysisArgs(self, args: AnalysisArgs):
        self._analysisArgs = args

    def analyze(self):
        if self.experiment is None:
            return
        self.experiment.substrate.reference.verify()
        self.experiment.substrate.verify()
        self.experiment.verify()
        expt_kind = self.experimentKind()

        data_path = self.analysisArgs().data_path
        image_path = self.analysisArgs().image_path
        video_path = self.analysisArgs().video_path
        fps = self.analysisArgs().fps

        # make image generator
        if (
            expt_kind == ExperimentKind.SingleImageExperiment
            or expt_kind == ExperimentKind.MultiImageExperiment
        ):
            img_gen = (cv2.imread(path) for path in self.paths)
            if fps is None:
                fps = 0
            h, w = cv2.imread(self.paths[0]).shape[:2]
            total = len(self.paths)
        elif expt_kind == ExperimentKind.VideoExperiment:
            (path,) = self.paths
            cap = cv2.VideoCapture(path)
            fnum = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img_gen = (cap.read()[1] for _ in range(fnum))
            fps = cap.get(cv2.CAP_PROP_FPS)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            total = fnum
        else:
            raise TypeError(f"Unsupported experiment kind: {expt_kind}")
        self.progressMaximumChanged.emit(total)

        # prepare for data writing
        if data_path:
            write_data = True
            dirname, _ = os.path.split(data_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            _, data_ext = os.path.splitext(data_path)
            data_ext = data_ext.lstrip(os.path.extsep).lower()
            writercls = self.data_writers.get(data_ext, None)
            if writercls is None:
                raise TypeError(f"Unsupported extension: {data_ext}")
            headers = [
                f.name for f in dataclasses.fields(self.experiment.layer_type.Data)
            ]
            if fps:
                headers = ["time (s)"] + headers
            datawriter = writercls(data_path, headers)
            datawriter.prepare()
        else:
            write_data = False

        # prepare for image writing
        if image_path:
            write_image = True
            dirname, _ = os.path.split(image_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            try:
                image_path % 0
                image_path_formattable = True
            except (TypeError, ValueError):
                image_path_formattable = False
        else:
            write_image = False

        # prepare for video writing
        if video_path:
            write_video = True
            _, video_ext = os.path.splitext(video_path)
            video_ext = video_ext.lstrip(os.path.extsep).lower()
            fourcc = self.video_codecs.get(video_ext, None)
            if fourcc is None:
                raise TypeError(f"Unsupported extension: {video_ext}")
            dirname, _ = os.path.split(video_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            videowriter = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        else:
            write_video = False

        # analyze!
        layer_gen = self.experiment.layer_generator()
        try:
            for i, img in enumerate(img_gen):
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                next(layer_gen)
                layer = layer_gen.send(img)
                valid = layer.valid()
                if write_data:
                    if valid:
                        data = list(dataclasses.astuple(layer.analyze()))
                        if fps:
                            data = [i / fps] + data
                    else:
                        data = []
                    datawriter.write_data(data)
                if write_image or write_video:
                    if valid:
                        visualized = layer.draw()
                    else:
                        visualized = img
                    visualized = cv2.cvtColor(visualized, cv2.COLOR_RGB2BGR)
                    if write_image:
                        if image_path_formattable:
                            imgpath = image_path % i
                        else:
                            imgpath = image_path
                        cv2.imwrite(imgpath, visualized)
                    if write_video:
                        videowriter.write(visualized)
                self.progressValueChanged.emit(i + 1)
        finally:
            if write_data:
                datawriter.terminate()
            if write_video:
                videowriter.release()

    def clear(self):
        self._experiment = None
        self._paths = []
        self._expt_kind = ExperimentKind.NullExperiment
        self._analysisArgs = AnalysisArgs()        
