.. _howto-runtime:

How to analyze an experiment in runtime
=======================================

.. currentmodule:: finitedepth

When :ref:`command-line analysis <basic-example>` is invoked, instances of
the following classes are constructed for the analysis:

#. :ref:`Reference class <howto-reference>`
#. :ref:`Substrate class <howto-substrate>`
#. :ref:`Coating layer class <howto-coatinglayer>`
#. :ref:`Experiment class <howto-experiment>`
#. :ref:`Analysis class <howto-analysis>`

In this page, detailed instructions to construct and use these objects in
runtime are provided.

.. _howto-reference:

Reference instance
------------------

Reference instance wraps *reference image*, which is a binary image of
a bare substrate.

The role of the reference instance is to specify regions in the
reference image which are used by substrate instance and
coating layer instance.

To construct reference instance, we must first read reference image.
:func:`get_data_path` is a handy function to access the sample data:

.. plot::
    :include-source:
    :context: reset

    >>> import cv2
    >>> from dipcoatimage.finitedepth import get_data_path
    >>> gray = cv2.imread(get_data_path("ref3.png"), cv2.IMREAD_GRAYSCALE)
    >>> _, refimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    >>> plt.imshow(refimg, cmap="gray")  #doctest: +SKIP

Now we can construct a reference instance. Type of the reference
instance must be a concrete subclass of :class:`ReferenceBase`. In
this example, we use :class:`Reference`.

.. plot::
    :include-source:
    :context: close-figs

    >>> from dipcoatimage.finitedepth import Reference
    >>> ref = Reference(
    ...     refimg,
    ...     templateROI=(200, 50, 1200, 200),
    ...     substrateROI=(350, 175, 1000, 500),
    ...     parameters=Reference.Parameters(),
    ...     draw_options=Reference.DrawOptions(),
    ... )
    >>> plt.imshow(ref.draw())  #doctest: +SKIP

Apart from the reference image, :class:`ReferenceBase` has
four more arguments which are consistent for every concrete
subclass.

#. *templateROI* (tuple)
    Specify a region which is used by coating layer instance
    to find the substrate location.
#. *substrateROI* (tuple)
    Specify a region which is used by substrate instance
    to analyze the substrate geometry.
#. *parameters* (:obj:`dataclass <dataclasses.dataclass>`)
    Any additional parameters which affect the analysis of
    reference instance.

    The type of the dataclass is defined in
    :attr:`~ReferenceBase.Parameters` attribute,
    which varies by each concrete subclass.
#. *draw_options* (:obj:`dataclass <dataclasses.dataclass>`)
    Options to visualize the reference instance.

    The type of the dataclass is defined in
    :attr:`~ReferenceBase.DrawOptions` attribute,
    which varies by each concrete subclass.

.. note::

    In order to use default values, no argument is passed to
    :attr:`~Reference.Parameters` and
    :attr:`~Reference.DrawOptions` in this
    example. Not passing the *parameters* and *draw_options*
    to :class:`Reference` at all has the same effect.

    .. code-block:: python

        ref = Reference(
            refimg,
            templateROI=(200, 50, 1200, 200),
            substrateROI=(350, 175, 1000, 500),
        )

    This will try to construct
    :attr:`~Reference.Parameters` and
    :attr:`~Reference.DrawOptions` with default
    values. Note that this will fail if the dataclasses
    themselves define required fields.

:meth:`~ReferenceBase.draw` method returns visualized result using
the options stored in :attr:`~ReferenceBase.draw_options`.
The options can be modified:

.. plot::
    :include-source:
    :context: close-figs

    >>> ref.draw_options.templateROI.color = (0, 0, 255)
    >>> plt.imshow(ref.draw())  #doctest: +SKIP

:meth:`~ReferenceBase.analyze` method returns numerical analysis
results wrapped in a dataclass. Its type is defined in
:attr:`~ReferenceBase.Data` attribute, which varies by each
concrete subclass.

The following returns empty data because :class:`Reference`
does not define any numerical analysis:

>>> ref.analyze()
Data()

.. _howto-substrate:

Substrate instance
------------------

Substrate instance wraps reference instance to get the substrate
region in reference image.

The role of the substrate instance is to analyze the geometry
of the bare substrate, which is used by coating layer instance
to quantify the shape of the coating layer.

Type of the substrate instance must be a concrete subclass of
:class:`SubstrateBase`. In this example, we use :class:`RectSubstrate`
which detects the corners and edges of rectangular substrate.

.. plot::
    :include-source:
    :context: close-figs

    >>> from dipcoatimage.finitedepth import RectSubstrate
    >>> subst = RectSubstrate(
    ...     ref,
    ...     parameters=RectSubstrate.Parameters(Sigma=3.0, Rho=1.0, Theta=0.01),
    ...     draw_options=RectSubstrate.DrawOptions(),
    ... )
    >>> plt.imshow(subst.draw())  #doctest: +SKIP

Similarly to the reference type, :class:`SubstrateBase` has strictly
defined signatures. Apart from the reference instance, we need two
more arguments:

#. *parameters* (:obj:`dataclass <dataclasses.dataclass>`)
    Any additional parameters which affect the analysis of
    substrate instance.

    The type of the dataclass is defined in
    :attr:`~SubstrateBase.Parameters` attribute,
    which varies by each concrete subclass.
#. *draw_options* (:obj:`dataclass <dataclasses.dataclass>`)
    Options to visualize the substrate instance.

    The type of the dataclass is defined in
    :attr:`~SubstrateBase.DrawOptions` attribute,
    which varies by each concrete subclass.

:meth:`~SubstrateBase.draw` method returns visualized result which
can be controlled by :attr:`~SubstrateBase.draw_options`, and
:meth:`~SubstrateBase.analyze` method returns numerical data.
Now you might be starting to see the repeating design pattern.

.. plot::
    :include-source:
    :context: close-figs

    >>> subst.analyze()
    Data(Width=525.98883)
    >>> subst.draw_options.vertices.linewidth = 3
    >>> plt.imshow(subst.draw())  #doctest: +SKIP

.. note::

    Unlike visualization result, the numerical result is strictly
    determined by immutable :attr:`SubstrateBase.parameters` and
    is never changed. This design allows caching of expensive
    intermediate steps, and applies to reference instance and
    coating layer instance as well.

.. _howto-coatinglayer:

Coating layer instance
----------------------

Coating layer instance wraps substrate instance and *target image*,
which is a binary image of a coated substrate.

Coating layer instance is the most important object in analysis. It extracts
the coating layer region from the target image and analyze it,
quantifying the shape of the coating layer.

We first read target image.

.. plot::
    :include-source:
    :context: close-figs

    >>> import cv2
    >>> from dipcoatimage.finitedepth import get_data_path
    >>> gray = cv2.imread(get_data_path("coat3.png"), cv2.IMREAD_GRAYSCALE)
    >>> _, coatimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    >>> plt.imshow(coatimg, cmap="gray")  #doctest: +SKIP

Now we construct a coating layer instance. Type of the coating layer instance
must be a concrete subclass of :class:`CoatingLayerBase`. In this example,
we use :class:`RectLayerShape` which analyzes the shape of the coating layer
over :class:`RectSubstrate`.

.. plot::
    :include-source:
    :context: close-figs

    >>> from dipcoatimage.finitedepth import RectLayerShape
    >>> layer_param=RectLayerShape.Parameters(
    ...     KernelSize=(1, 1),
    ...     ReconstructRadius=50,
    ...     RoughnessMeasure=RectLayerShape.DistanceMeasure.DTW,
    ... )
    >>> coat = RectLayerShape(
    ...     coatimg,
    ...     subst,
    ...     parameters=layer_param,
    ...     draw_options=RectLayerShape.DrawOptions(),
    ...     deco_options=RectLayerShape.DecoOptions(),
    ... )
    >>> plt.imshow(coat.draw())  #doctest: +SKIP

:class:`CoatingLayerBase` has somewhat more complicated signatures.
Apart from the target image and substrate instance, there are four
more arguments:

#. *parameters* (:obj:`dataclass <dataclasses.dataclass>`)
    Any additional parameters which affect the analysis of
    coating layer instance.

    The type of the dataclass is defined in
    :attr:`~CoatingLayerBase.Parameters` attribute,
    which varies by each concrete subclass.
#. *draw_options* (:obj:`dataclass <dataclasses.dataclass>`)
    Options to visualize the coated substrate.

    The type of the dataclass is defined in
    :attr:`~CoatingLayerBase.DrawOptions` attribute,
    which varies by each concrete subclass.
#. *deco_options* (:obj:`dataclass <dataclasses.dataclass>`)
    Options to visualize the coating layer.

    The type of the dataclass is defined in
    :attr:`~CoatingLayerBase.DecoOptions` attribute,
    which varies by each concrete subclass.
#. *tempmatch* (tuple)
    Location of the template region in the target image.

    **NOT FOR DIRECT USE.** This argument is for experiment
    instance to quickly construct coating layer instances,
    which is described in the next section. If not passed,
    the coating layer instance automatically finds the template
    location using template matching.

:meth:`~CoatingLayerBase.draw` method returns visualized result which
can be controlled by :attr:`~CoatingLayerBase.draw_options` and
:attr:`~CoatingLayerBase.deco_options`. :meth:`~CoatingLayerBase.analyze`
method returns numerical data.

.. plot::
    :include-source:
    :context: close-figs

    >>> coat.analyze()
    Data(LayerLength_Left=236.6932474452997, ...)
    >>> coat.draw_options.subtraction = coat.SubtractionMode.TEMPLATE
    >>> coat.deco_options.roughness.linewidth = 0
    >>> plt.imshow(coat.draw())  #doctest: +SKIP

.. note::

    *draw_options* and *deco_options* are designed to cover different scopes.
    In general, *draw_options* controls how the substrate body is drawn,
    while *deco_options* controls how the analysis result is displayed.

Typically, multiple coating layer instances are constructed from consecutive
target images for temporal evaluation. Experiment instance facilitates the
sequential construction of coating layer instances.

.. _howto-experiment:

Experiment instance
-------------------

Experiment instance is a coating layer instance factory.

Why the name *experiment*? It's because conceptually, conducting an experiment
is acquiring useful data from coated substrate. Experiment instance defines how
consecutive target images, rather than a single image, construct coating layer
instances.

Type of the experiment instance must be a concrete subclass of
:class:`ExperimentBase`. In this example, we use :class:`Experiment` which
narrows down the template location using the previous result, significantly
speeding up the construction.

.. plot::
    :include-source:
    :context: close-figs

    >>> from dipcoatimage.finitedepth import Experiment
    >>> expt = Experiment(
    ...     parameters=Experiment.Parameters(window=(5, 5))
    ... )

:class:`ExperimentBase` has only one argument *parameters*, whose type is
defined in :attr:`~ExperimentBase.Parameters` attribute.

Now, we sequentially construct two :class:`RectLayerShape` instances using same
arguments. The result is identical, but the second construction is several
times faster because template matching is performed over only a small window.

.. plot::
    :include-source:
    :context: close-figs

    >>> coat1 = expt.coatinglayer(coatimg, subst, RectLayerShape, layer_param)
    >>> coat2 = expt.coatinglayer(coatimg, subst, RectLayerShape, layer_param)
    >>> _, axes = plt.subplots(1, 2)  #doctest: +SKIP
    >>> axes[0].imshow(coat1.draw())  #doctest: +SKIP
    >>> axes[1].imshow(coat2.draw())  #doctest: +SKIP
    >>> plt.show()  #doctest: +SKIP

.. _howto-analysis:

Analysis instance
-----------------

Analysis instance is a coroutine which receives the coating layer instances
to save the analysis result as file.

Type of the analysis instance must be a concrete subclass of
:class:`AnalysisBase`. In this example, we use :class:`Analysis`.

>>> from dipcoatimage.finitedepth import Analysis
>>> analysis = Analysis(
...     parameters=Analysis.Parameters(
...         layer_visual="result%d.jpg",
...         layer_data="result.csv",
... ),
...     fps=50,
... )

#. *parameters* (:obj:`dataclass <dataclasses.dataclass>`)
    Any additional parameters which affect the resulting file

    The type of the dataclass is defined in
    :attr:`~AnalysisBase.Parameters` attribute,
    which varies by each concrete subclass.
#. *fps* (float, optional)
    Frame rate per second of the target images.

    This argument is required to write videos and mark the data with timestamps.

You can now start the coroutine, send the coating layer instances as much as
you want, and close it.

>>> analysis.send(None)  # start the coroutine  #doctest: +SKIP
>>> analysis.send(coat1)  #doctest: +SKIP
>>> analysis.send(coat2)  #doctest: +SKIP
>>> analysis.close()  #doctest: +SKIP

Congratulations! You have successfully performed analysis.

Configuration instance
----------------------

Reading the images and constructing all the instances is a tedious work.
Configuration instance helps you automatize essentially every task that have been
described so far:

#. Read the reference image and target images.
#. Binarize the images.
#. Construct the instances.
#. Determine the FPS of target images, if possible.
#. Start and finish the analysis coroutine.

Type of the configuration instance must be a concrete subclass of
:class:`ConfigBase`. In this example, we use :class:`Config` to analyze the
target images in a video:

>>> from dipcoatimage.finitedepth.serialize import *
>>> config = Config(
...     ref_path=get_data_path("ref3.png"),
...     coat_path=get_data_path("coat3.mp4"),
...     reference=ReferenceArgs(
...         templateROI=(200, 50, 1200, 200),
...         substrateROI=(350, 175, 1000, 500),
...     ),
...     substrate=SubstrateArgs(
...         type=ImportArgs(name="RectSubstrate"),
...         parameters=dict(Sigma=3.0, Rho=1.0, Theta=0.01),
...     ),
...     coatinglayer=CoatingLayerArgs(
...         type=ImportArgs(name="RectLayerShape"),
...         parameters=dict(
...             KernelSize=(1, 1),
...             ReconstructRadius=50,
...             RoughnessMeasure=RectLayerShape.DistanceMeasure.DTW,
...         ),
...     ),
...     experiment=ExperimentArgs(
...         parameters=dict(window=(5, 5)),
...     ),
...     analysis=AnalysisArgs(
...         parameters=dict(layer_visual="result%d.jpg", layer_data="result.csv")
...     ),
... )
>>> config.analyze("My analysis")  #doctest: +SKIP

As you can see, passing every argument to :class:`Config` is quite verbose.
A better approach is to write a :ref:`configuration file <config-reference>`,
read it as :obj:`dict`, and constructing the configuration instance from it.

As :class:`ConfigBase` is a dataclass, it can be easily structured using
:mod:`catters` package. Use :obj:`data_converter` to convert the dictionary
to configuration intance (and vice versa).

.. code-block:: python

    import yaml
    from dipcoatimage.finitedepth import data_converter
    with open("my-config-file.yml") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in data.items():
        config = data_converter(v, Config)
        config.analyze(k)

See :ref:`tutorial` for configuration file examples.
