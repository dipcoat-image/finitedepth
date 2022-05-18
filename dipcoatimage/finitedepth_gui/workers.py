"""
Workers
=======

This module provides workers to construct objects from the parameters passed to
control widgets.

"""

import cv2  # type: ignore
from dipcoatimage.finitedepth import (
    SubstrateReferenceBase,
    SubstrateBase,
    CoatingLayerBase,
    ExperimentBase,
)
import dataclasses
from dipcoatimage.finitedepth.analysis import (
    AnalysisArgs,
    experiment_kind,
    ExperimentKind,
)
from dipcoatimage.finitedepth.util import OptionalROI, DataclassProtocol
import enum
import numpy as np
import numpy.typing as npt
import os
from PySide6.QtCore import QObject, QModelIndex, Slot, Signal, Qt
from PySide6.QtGui import QStandardItem
from typing import Optional, Type, Generator, List
from .core import (
    StructuredReferenceArgs,
    StructuredSubstrateArgs,
    StructuredCoatingLayerArgs,
    StructuredExperimentArgs,
)
from .inventory import (
    ExperimentItemModel,
)


__all__ = [
    "VisualizationMode",
    "WorkerBase",
    "ReferenceWorker",
    "SubstrateWorker",
    "ExperimentWorker",
    "AnalysisWorker",
    "MasterWorker",
]


class VisualizationMode(enum.IntEnum):
    """
    Option for workers to determine how the image is shown.

    Attributes
    ==========

    OFF
        Do not visualize.

    FULL
        Full visualization. Reference and substrate are visualized as usual, and
        coating layer is visualized using coating layer decoration.

    FAST
        Fast visualization without coating layer decoration. Reference and
        substrate are visualized as usual, but coating layer is not decorated.

    """

    OFF = 0
    FULL = 1
    FAST = 2


class WorkerBase(QObject):
    """Base class for all worker objects."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._visualize_mode = VisualizationMode.FULL

    def visualizationMode(self) -> VisualizationMode:
        return self._visualize_mode

    def setVisualizationMode(self, mode: VisualizationMode):
        self._visualize_mode = mode


class ReferenceWorker(WorkerBase):
    """
    Worker to build the concreate instance of :class:`SubstrateReferenceBase`
    and to visualize it.

    Data for reference object are:

    1. :meth:`referenceType`
    2. :meth:`image`
    3. :meth:`templateROI`
    4. :meth:`substrateROI`
    5. :meth:`paramters`
    6. :meth:`drawOptions`

    :meth:`image` is updated by :meth:`setImage`. Other data are updated by
    :meth:`setStructuredReferenceArgs`.

    :meth:`updateReference` constructs the reference object with current data.
    Resulting object can be acquired from :meth:`reference`, or calling
    :meth:`emitReference` and listening to :attr:`referenceChanged` signal.

    Visualization result can be directly acquired from :meth:`visualizedImage`.

    """

    referenceChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._type = None
        self._img = None
        self._temproi = (0, 0, None, None)
        self._substroi = (0, 0, None, None)
        self._params = None
        self._draw_opts = None

        self._reference = None

    def referenceType(self) -> Optional[Type[SubstrateReferenceBase]]:
        """
        Type object to construct :meth:`reference`.
        ``None`` indicates invalid value.
        """
        return self._type

    def image(self) -> npt.NDArray[np.uint8]:
        """*image* for :meth:`referenceType` to construct :meth:`reference`."""
        img = self._img
        if img is None:
            img = np.empty((0, 0, 0), dtype=np.uint8)
        return img

    def setImage(self, img: Optional[npt.NDArray[np.uint8]]):
        """
        Set :meth:`image` with *img*.

        This does not update :meth:`reference`. Run :meth:`updateReference`
        manually.
        """
        self._img = img

    def templateROI(self) -> OptionalROI:
        """
        *templateROI* for :meth:`referenceType` to construct :meth:`reference`.
        """
        return self._temproi

    def substrateROI(self) -> OptionalROI:
        """
        *substrateROI* for :meth:`referenceType` to construct :meth:`reference`.
        """
        return self._substroi

    def parameters(self) -> Optional[DataclassProtocol]:
        """
        *parameters* for :meth:`referenceType` to construct :meth:`reference`.
        ``None`` indicates invalid value.
        """
        return self._params

    def drawOptions(self) -> Optional[DataclassProtocol]:
        """
        *draw_options* for :meth:`referenceType` to construct :meth:`reference`.
        ``None`` indicates invalid value.
        """
        return self._draw_opts

    def setStructuredReferenceArgs(self, data: StructuredReferenceArgs):
        """
        Set following values with *data*.

        1. :meth:`referenceType`
        2. :meth:`image`
        3. :meth:`templateROI`
        4. :meth:`substrateROI`
        5. :meth:`paramters`
        6. :meth:`drawOptions`

        This does not update :meth:`reference`. Run :meth:`updateReference`
        manually.
        """
        reftype = data.type
        if not (
            isinstance(reftype, type) and issubclass(reftype, SubstrateReferenceBase)
        ):
            reftype = None
        self._type = reftype

        self._temproi = data.templateROI
        self._substroi = data.substrateROI

        params = data.parameters
        if reftype is None:
            params = None
        elif isinstance(params, reftype.Parameters):
            pass
        else:
            try:
                params = reftype.Parameters()
            except TypeError:
                params = None
        self._params = params

        drawopt = data.draw_options
        if reftype is None:
            drawopt = None
        elif isinstance(drawopt, reftype.DrawOptions):
            pass
        else:
            try:
                drawopt = reftype.DrawOptions()
            except TypeError:
                drawopt = None
        self._draw_opts = drawopt

    def updateReference(self):
        """Update :meth:`reference` and emit to :attr:`referenceChanged`."""
        ref = None
        default_invalid_args = [
            self.referenceType(),
            self.parameters(),
            self.drawOptions(),
        ]
        if all(x is not None for x in default_invalid_args) and self.image().size > 0:
            ref = self.referenceType()(
                self.image(),
                self.templateROI(),
                self.substrateROI(),
                parameters=self.parameters(),
                draw_options=self.drawOptions(),
            )
            if not ref.valid():
                ref = None
        self._reference = ref
        self.emitReference()

    def reference(self) -> Optional[SubstrateReferenceBase]:
        """
        Concrete instance of :class:`SubstrateReferenceBase`.
        ``None`` indicates invalid value.

        Run :meth:`updateReference` to update this value.
        """
        return self._reference

    def emitReference(self):
        """Emit the result of :meth:`reference` to :attr:`referenceChanged`."""
        self.referenceChanged.emit(self.reference())

    def visualizedImage(self) -> npt.NDArray[np.uint8]:
        """
        Return visualization result of :meth:`reference`.

        If :meth:`reference` is invalid or :meth:`visualizationMode` is False,
        directly return :meth:`image`.
        """
        ref = self.reference()
        if ref is not None and self.visualizationMode():
            image = ref.draw()
        else:
            image = self.image()
        return image


class SubstrateWorker(WorkerBase):
    """
    Worker to build the concreate instance of :class:`SubstrateBase` and to
    visualize it.

    Data for substrate object are:

    1. :meth:`substrateType`
    2. :meth:`reference`
    3. :meth:`paramters`
    4. :meth:`drawOptions`

    :meth:`reference` is updated by :meth:`setReference`, and other data are
    updated by :meth:`setStructuredSubstrateArgs`.

    :meth:`updateSubstrate` constructs the substrate object with current data.
    Resulting object can be acquired from :meth:`substrate`, or calling
    :meth:`emitSubstrate` and listening to :attr:`substrateChanged` signal.

    Visualization result can be directly acquired from :meth:`visualizedImage`.

    """

    substrateChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._type = None
        self._ref = None
        self._params = None
        self._draw_opts = None

        self._substrate = None

    def substrateType(self) -> Optional[Type[SubstrateBase]]:
        """
        Type object to construct :meth:`substrate`.
        ``None`` indicates invalid value.
        """
        return self._type

    def reference(self) -> Optional[SubstrateReferenceBase]:
        """
        Substrate reference instance to construct :meth:`substrate`.
        ``None`` indicates invalid value.
        """
        return self._ref

    def parameters(self) -> Optional[DataclassProtocol]:
        """
        *parameters* for :meth:`substrateType` to construct :meth:`substrate`.
        ``None`` indicates invalid value.
        """
        return self._params

    def drawOptions(self) -> Optional[DataclassProtocol]:
        """
        *draw_options* for :meth:`substrateType` to construct :meth:`substrate`.
        ``None`` indicates invalid value.
        """
        return self._draw_opts

    def setStructuredSubstrateArgs(self, data: StructuredSubstrateArgs):
        """
        Set following values with *data*.

        1. :meth:`substrateType`
        2. :meth:`paramters`
        3. :meth:`drawOptions`

        This does not update :meth:`substrate`. Run :meth:`updateSubstrate`
        manually.
        """
        substtype = data.type
        if not (isinstance(substtype, type) and issubclass(substtype, SubstrateBase)):
            substtype = None
        self._type = substtype

        params = data.parameters
        if substtype is None:
            params = None
        elif isinstance(params, substtype.Parameters):
            pass
        else:
            try:
                params = substtype.Parameters()
            except TypeError:
                params = None
        self._params = params

        drawopt = data.draw_options
        if substtype is None:
            drawopt = None
        elif isinstance(drawopt, substtype.DrawOptions):
            pass
        else:
            try:
                drawopt = substtype.DrawOptions()
            except TypeError:
                drawopt = None
        self._draw_opts = drawopt

    @Slot(object)
    def setReference(self, ref: Optional[SubstrateReferenceBase]):
        """
        Set :meth:`reference` with *ref*.

        This does not update :meth:`substrate`. Run :meth:`updateSubstrate`
        manually.
        """
        self._ref = ref

    def updateSubstrate(self):
        """Update :meth:`substrate` and emit to :attr:`substrateChanged`."""
        subst = None
        default_invalid_args = [
            self.substrateType(),
            self.reference(),
            self.parameters(),
            self.drawOptions(),
        ]
        if all(x is not None for x in default_invalid_args):
            subst = self.substrateType()(
                self.reference(),
                parameters=self.parameters(),
                draw_options=self.drawOptions(),
            )
            if not subst.valid():
                subst = None
        self._substrate = subst
        self.emitSubstrate()

    def substrate(self) -> Optional[SubstrateBase]:
        """
        Concrete instance of :class:`SubstrateBase`.
        ``None`` indicates invalid value.

        Run :meth:`updateSubstrate` to update this value.
        """
        return self._substrate

    def emitSubstrate(self):
        """Emit the result of :meth:`substrate` to :attr:`substrateChanged`."""
        self.substrateChanged.emit(self.substrate())

    def visualizedImage(self) -> npt.NDArray[np.uint8]:
        """
        Return visualization result of :meth:`substrate`.

        If :meth:`substrate` is invalid or :meth:`visualizationMode` is False,
        directly return :meth:`image`.
        """
        subst = self.substrate()
        ref = self.reference()
        if subst is not None:
            if self.visualizationMode():
                image = subst.draw()
            else:
                image = subst.image()
        elif ref is not None:
            image = ref.substrate_image()
        else:
            image = np.empty((0, 0, 0), dtype=np.uint8)
        return image


class ExperimentWorker(WorkerBase):
    """
    Worker to build the concreate instance of :class:`ExperimentBase` and to
    visualize it.

    Data for experiment object are:

    1. :meth:`experimentType`
    2. :meth:`substrate`
    3. :meth:`coatingLayerType`
    4. :meth:`coatingLayerParameters`
    5. :meth:`coatingLayerDrawOptions`
    6. :meth:`coatingLayerDecoOptions`
    7. :meth:`parameters`

    :meth:`substrate` is updated by :meth:`setSubstrate`, and other data are
    updated by :meth:`setStructuredCoatingLayerArgs`. and
    :meth:`setStructuredExperimentArgs`

    :meth:`updateExperiment` constructs the experiment object with data.
    Resulting object can be acquired by :meth:`experiment`, or calling
    :meth:`emitExperiment` and listening to :attr:`experimentChanged` signal.

    To visualize the layer shape image, pass it to :meth:`setImage` first.
    Visualization result can be directly acquired from:meth:`visualizedImage`.

    """

    experimentChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._type = None
        self._subst = None
        self._layer_type = None
        self._layer_params = None
        self._layer_drawopts = None
        self._layer_decoopts = None
        self._params = None

        self._expt = None
        self._layer_generator = None

        self._img = None

    def experimentType(self) -> Optional[Type[ExperimentBase]]:
        """
        Type object to construct :meth:`experiment`. ``None`` indicates invalid
        value.
        """
        return self._type

    def substrate(self) -> Optional[SubstrateBase]:
        """
        Substrate instance to construct :meth:`experiment`. ``None`` indicates
        invalid value.
        """
        return self._subst

    def coatingLayerType(self) -> Optional[Type[CoatingLayerBase]]:
        """
        Coating layer type object to construct :meth:`experiment`. ``None``
        indicates invalid value.
        """
        return self._layer_type

    def coatingLayerParameters(self) -> Optional[DataclassProtocol]:
        """
        *parameters* for :meth:`coatingLayerType` to construct
        :meth:`experiment`. ``None`` indicates invalid value.
        """
        return self._layer_params

    def coatingLayerDrawOptions(self) -> Optional[DataclassProtocol]:
        """
        *draw_options* for :meth:`coatingLayerType` to construct
        :meth:`experiment`. ``None`` indicates invalid value.
        """
        return self._layer_drawopts

    def coatingLayerDecoOptions(self) -> Optional[DataclassProtocol]:
        """
        *deco_options* for :meth:`coatingLayerType` to construct
        :meth:`experiment`. ``None`` indicates invalid value.
        """
        return self._layer_decoopts

    def parameters(self) -> Optional[DataclassProtocol]:
        """
        *parameters* for :meth:`experimentType` to construct :meth:`experiment`.
        ``None`` indicates invalid value.
        """
        return self._params

    @Slot(object)
    def setSubstrate(self, subst: Optional[SubstrateBase]):
        """
        Set :meth:`substrate` with *subst*.

        This does not update :meth:`experiment`. Run :meth:`updateExperiment`
        manually.
        """
        self._subst = subst

    def setStructuredCoatingLayerArgs(self, data: StructuredCoatingLayerArgs):
        """
        Set following values with *data*.

        1. :meth:`coatingLayerType`
        2. :meth:`coatingLayerParameters`
        3. :meth:`coatingLayerDrawOptions`
        4. :meth:`coatingLayerDecoOptions`

        This does not update :meth:`experiment`. Run :meth:`updateExperiment`
        manually.
        """
        coattype = data.type

        if not (isinstance(coattype, type) and issubclass(coattype, CoatingLayerBase)):
            coattype = None
        self._layer_type = coattype

        coat_params = data.parameters
        if coattype is None:
            coat_params = None
        elif isinstance(coat_params, coattype.Parameters):
            pass
        else:
            try:
                coat_params = coattype.Parameters()
            except TypeError:
                coat_params = None
        self._layer_params = coat_params

        coat_drawopts = data.draw_options
        if coattype is None:
            coat_drawopts = None
        elif isinstance(coat_drawopts, coattype.DrawOptions):
            pass
        else:
            try:
                coat_drawopts = coattype.DrawOptions()
            except TypeError:
                coat_drawopts = None
        self._layer_drawopts = coat_drawopts

    def setStructuredExperimentArgs(self, data: StructuredExperimentArgs):
        """
        Set following values with *data*.

        1. :meth:`experimentType`
        2. :meth:`experimentParameters`

        This does not update :meth:`experiment`. Run :meth:`updateExperiment`
        manually.
        """
        expttype = data.type
        if not (isinstance(expttype, type) and issubclass(expttype, ExperimentBase)):
            expttype = None
        self._type = expttype

        params = data.parameters
        if expttype is None:
            params = None
        elif isinstance(params, expttype.Parameters):
            pass
        else:
            try:
                params = expttype.Parameters()
            except TypeError:
                params = None
        self._params = params

    def updateExperiment(self):
        """
        Update :meth:`experiment` and emit to :attr:`experimentChanged`, and
        update :meth:`layerGenerator`.
        """
        expt = None

        default_invalid_args = [
            self.experimentType(),
            self.substrate(),
            self.coatingLayerType(),
            self.coatingLayerParameters(),
            self.coatingLayerDrawOptions(),
            self.coatingLayerDecoOptions(),
            self.parameters(),
        ]
        if all(x is not None for x in default_invalid_args):
            expt = self.experimentType()(
                self.substrate(),
                self.coatingLayerType(),
                self.coatingLayerParameters(),
                self.coatingLayerDrawOptions(),
                self.coatingLayerDecoOptions(),
                parameters=self.parameters(),
            )
            if not expt.valid():
                expt = None
        self._expt = expt
        self.updateLayerGenerator()
        self.emitExperiment()

    def experiment(self) -> Optional[ExperimentBase]:
        """
        Concrete instance of :class:`ExperimentBase` constructed by experiment
        object data. ``None`` indicates invalid value.

        Run :meth:`updateExperiment` to update this value.
        """
        return self._expt

    def emitExperiment(self):
        """Emit the result of :meth:`experiment` to :attr:`experimentChanged`."""
        self.experimentChanged.emit(self.experiment())

    @Slot()
    def updateLayerGenerator(self):
        expt = self.experiment()
        if expt is not None:
            self._layer_generator = expt.layer_generator()
            next(self.layerGenerator())
        else:
            self._layer_generator = None

    def layerGenerator(
        self,
    ) -> Generator[CoatingLayerBase, npt.NDArray[np.uint8], None]:
        """
        :meth:`Experiment.layer_generator` from :meth:`experiment`.
        ``None`` indicates invalid value.

        Run :meth:`updateLayerGenerator` to update this value.
        """
        return self._layer_generator

    def setVisualizationMode(self, mode):
        """
        Update :meth:`visualizationMode` with *mode*.
        Also, update :meth:`layerGenerator` if *mode* is True.
        """
        super().setVisualizationMode(mode)
        if mode:
            self.updateLayerGenerator()

    def image(self) -> npt.NDArray[np.uint8]:
        """Layer shape image to be visualized."""
        img = self._img
        if img is None:
            img = np.empty((0, 0, 0))
        return img

    def setImage(self, img: Optional[npt.NDArray[np.uint8]]):
        """Update :meth:`image` with *img*."""
        self._img = img

    def visualizedImage(self) -> npt.NDArray[np.uint8]:
        """
        Return visualization result of :meth:`image` analyzed by
        :meth:`experiment`.

        If possible, :meth:`ExperimentBase.layer_generator` is used to construct
        the visualized image. This implies that consecutively visualized images
        are consecutive in real world as well. If completely unrelated images
        should be passed, run :meth:`updateLayerGenerator` first.

        If :meth:`visualizationMode` is ``FULL``, visualization is done by
        constructing analysis objects. If it is ``FAST``, image is visualized by
        :meth:`fastVisualize`.

        If parameters are invalid or :meth:`visualizationMode` is ``OFF``,
        directly return :meth:`image`.

        """
        ls_gen = self.layerGenerator()
        vismode = self.visualizationMode()
        image = self.image()
        if ls_gen is not None and image.size > 0:
            if vismode == VisualizationMode.FULL:
                ls = ls_gen.send(image)
                while ls is None:  # previous construction not finished
                    ls = ls_gen.send(image)
                image = ls.draw()
            elif vismode == VisualizationMode.FAST:
                image = self.fastVisualize()
            else:
                pass
        return image

    def fastVisualize(self) -> npt.NDArray[np.uint8]:
        """
        Remove substrate from :meth:`image` without constructing analysis
        objects.

        This visualization lacks the versatility of full visualization, but it is
        considerably faster for quick overview.
        """
        substrate = self.substrate()
        if substrate is None:
            return self.image()

        x0, y0, x1, y1 = substrate.reference.templateROI
        template = substrate.reference.image[y0:y1, x0:x1]
        res = cv2.matchTemplate(self.image(), template, cv2.TM_CCOEFF)
        _, _, _, (tx, ty) = cv2.minMaxLoc(res)
        dx, dy = substrate.reference.temp2subst()
        x0, y0 = (tx + dx, ty + dy)
        subst_h, subst_w = substrate.image().shape[:2]
        img_h, img_w = self.image().shape[:2]
        x1, y1 = (x0 + subst_w, y0 + subst_h)

        _, bin_img = cv2.threshold(
            self.image(), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        bin_img_cropped = bin_img[
            max(y0, 0) : min(y1, img_h), max(x0, 0) : min(x1, img_w)
        ]
        subst_cropped = substrate.image()[
            max(-y0, 0) : min(img_h - y0, subst_h),
            max(-x0, 0) : min(img_w - x0, subst_w),
        ]
        _, bin_subst_cropped = cv2.threshold(
            subst_cropped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        xor = cv2.bitwise_xor(bin_img_cropped, bin_subst_cropped)
        nxor = cv2.bitwise_not(xor)
        bin_img[max(y0, 0) : min(y1, img_h), max(x0, 0) : min(x1, img_w)] = nxor
        return cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)


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
        self._analysisArgs = AnalysisArgs()

    def experiment(self) -> Optional[ExperimentBase]:
        return self._experiment

    def paths(self) -> List[str]:
        return self._paths

    def analysisArgs(self) -> AnalysisArgs:
        return self._analysisArgs

    @Slot(object)
    def setExperiment(self, expt: Optional[ExperimentBase]):
        self._experiment = expt

    def setPaths(self, paths: List[str]):
        self._paths = paths

    def setAnalysisArgs(self, args: AnalysisArgs):
        self._analysisArgs = args

    def analyze(self):
        if self.experiment is None:
            return
        self.experiment.substrate.reference.verify()
        self.experiment.substrate.verify()
        self.experiment.verify()
        expt_kind = experiment_kind(self.paths)

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


class MasterWorker(QObject):
    """
    Object which contains subworkers.
    """

    visualizedImageChanged = Signal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._exptitem_model = ExperimentItemModel()

        self._ref_worker = ReferenceWorker()
        self._subst_worker = SubstrateWorker()
        self._expt_worker = ExperimentWorker()
        self._anal_worker = AnalysisWorker()
        self._visualizing_worker = ExperimentWorker()

        self.connectModelSignals()
        self.referenceWorker().referenceChanged.connect(
            self.substrateWorker().setReference
        )
        self.substrateWorker().substrateChanged.connect(
            self.experimentWorker().setSubstrate
        )
        self.experimentWorker().experimentChanged.connect(
            self.analysisWorker().setExperiment
        )

    def experimentItemModel(self) -> ExperimentItemModel:
        """Model which holds the experiment item data."""
        return self._exptitem_model

    def referenceWorker(self) -> ReferenceWorker:
        return self._ref_worker

    def substrateWorker(self) -> SubstrateWorker:
        return self._subst_worker

    def experimentWorker(self) -> ExperimentWorker:
        return self._expt_worker

    def analysisWorker(self) -> AnalysisWorker:
        return self._anal_worker

    def visualizingWorker(self) -> WorkerBase:
        return self._visualizing_worker

    def setExperimentItemModel(self, model: ExperimentItemModel):
        """Set :meth:`experimentItemModel`."""
        self.disconnectModelSignals()
        self._exptitem_model = model
        self.connectModelSignals()

    def connectModelSignals(self):
        self.experimentItemModel().itemChanged.connect(self.onExperimentItemChange)
        self.experimentItemModel().rowsInserted.connect(self.onExperimentItemRowsChange)
        self.experimentItemModel().rowsRemoved.connect(self.onExperimentItemRowsChange)

    def disconnectModelSignals(self):
        self.experimentItemModel().itemChanged.disconnect(self.onExperimentItemChange)
        self.experimentItemModel().rowsInserted.disconnect(
            self.onExperimentItemRowsChange
        )
        self.experimentItemModel().rowsRemoved.disconnect(
            self.onExperimentItemRowsChange
        )

    @Slot(QStandardItem)
    def onExperimentItemChange(self, item: QStandardItem):
        if item.model() == self.experimentItemModel() and item.parent() is None:
            if item.column() == ExperimentItemModel.Col_Reference:
                data = self.experimentItemModel().data(item.index(), Qt.UserRole)[1]
                self.referenceWorker().setStructuredReferenceArgs(data)
                self.referenceWorker().updateReference()
            elif item.column() == ExperimentItemModel.Col_Substrate:
                data = self.experimentItemModel().data(item.index(), Qt.UserRole)[1]
                self.substrateWorker().setStructuredSubstrateArgs(data)
                self.substrateWorker().updateSubstrate()
            elif item.column() == ExperimentItemModel.Col_CoatingLayer:
                data = self.experimentItemModel().data(item.index(), Qt.UserRole)[1]
                self.experimentWorker().setStructuredCoatingLayerArgs(data)
                self.experimentWorker().updateExperiment()
            elif item.column() == ExperimentItemModel.Col_Experiment:
                data = self.experimentItemModel().data(item.index(), Qt.UserRole)[1]
                self.experimentWorker().setStructuredExperimentArgs(data)
                self.experimentWorker().updateExperiment()
        self.emitImage()

    @Slot(QModelIndex, int, int)
    def onExperimentItemRowsChange(self, index: QModelIndex, first: int, last: int):
        pass

    @Slot(QModelIndex)
    def setCurrentExperimentIndex(self, index: QModelIndex):
        """Set currently activated index from :meth:`experimentItemModel`."""
        if index.parent().isValid():
            raise TypeError("Only top-level index can be activated.")
        model = self.experimentItemModel()
        refpath = model.data(
            model.index(index.row(), ExperimentItemModel.Col_ReferencePath)
        )
        img = cv2.imread(refpath)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.referenceWorker().setImage(img)
        refargs = model.data(
            model.index(index.row(), ExperimentItemModel.Col_Reference),
            Qt.UserRole,
        )[1]
        self.referenceWorker().setStructuredReferenceArgs(refargs)
        substargs = model.data(
            model.index(index.row(), ExperimentItemModel.Col_Substrate),
            Qt.UserRole,
        )[1]
        self.substrateWorker().setStructuredSubstrateArgs(substargs)
        layerargs = model.data(
            model.index(index.row(), ExperimentItemModel.Col_CoatingLayer),
            Qt.UserRole,
        )[1]
        self.experimentWorker().setStructuredCoatingLayerArgs(layerargs)
        exptargs = model.data(
            model.index(index.row(), ExperimentItemModel.Col_Experiment),
            Qt.UserRole,
        )[1]
        self.experimentWorker().setStructuredExperimentArgs(exptargs)

        self.referenceWorker().updateReference()
        self.substrateWorker().updateSubstrate()
        self.experimentWorker().updateExperiment()
        self.emitImage()

    @Slot(object)
    def setReferenceImage(self, img: Optional[npt.NDArray[np.uint8]]):
        self.referenceWorker().setImage(img)
        self.referenceWorker().updateReference()
        self.substrateWorker().updateSubstrate()
        self.experimentWorker().updateExperiment()
        self.emitImage()

    def setVisualizingWorker(self, worker: WorkerBase):
        self._visualizing_worker = worker

    def emitImage(self):
        img = self.visualizingWorker().visualizedImage()
        self.visualizedImageChanged.emit(img)
