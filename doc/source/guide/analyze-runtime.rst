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
    >>> plt.imshow(refimg, cmap="gray") #doctest: +SKIP

Now we can construct a reference instance. Type of the reference
instance must be a concrete subclass of :class:`ReferenceBase`. In
this example, we use :class:`Reference`.

.. plot::
    :include-source:
    :context:

    >>> from dipcoatimage.finitedepth import Reference
    >>> ref = Reference(
    ...     refimg,
    ...     templateROI=(200, 50, 1200, 200),
    ...     substrateROI=(350, 175, 1000, 500),
    ...     parameters=Reference.Parameters(),
    ...     draw_options=Reference.DrawOptions(),
    ... )
    >>> plt.imshow(ref.draw()) #doctest: +SKIP

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
    :context:

    >>> ref.draw_options.templateROI.color = (0, 0, 255)
    >>> plt.imshow(ref.draw()) #doctest: +SKIP

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

Substrate instance handles substrate region in bare substrate image,
which is defined by reference instance.

.. _howto-coatinglayer:

Coating layer instance
----------------------

Coating layer instance handles *target image*, which is an image of
coated substrate.

.. _howto-experiment:

Experiment instance
-------------------

Experiment instance is a factory object to construct consecutive
coating layer instances.

.. _howto-analysis:

Analysis instance
-----------------

Analysis instance provides coroutine to store the analysis result
as files.

Using configuration file
------------------------

Analyze!
--------

Analysis is performed by constructing the coating layer instances
(preferably using experiment instance) and sending them to the
analysis coroutine,
