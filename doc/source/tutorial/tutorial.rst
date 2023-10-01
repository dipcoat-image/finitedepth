.. _tutorial:

Tutorial
========

.. currentmodule:: finitedepth

In this page, a step-by-step guide to achieve basic understanding of the
package is provided.

.. note::

    Before running this tutorial, environment variable ``$FINITEDEPTH_DATA``
    must be set:

    .. tabs::

        .. code-tab:: bash

            export FINITEDEPTH_DATA=$(finitedepth data)

        .. code-tab:: bat cmd

            FOR /F %G IN ('finitedepth data') DO SET FINITEDEPTH_DATA=%G

        .. code-tab:: powershell

            $env:FINITEDEPTH_DATA=$(finitedepth data)

    Check if the variable is properly set.
    The output of ``finitedepth data`` command should be same as the result of:

    .. tabs::

        .. code-tab:: bash

            echo $FINITEDEPTH_DATA

        .. code-tab:: bat cmd

            echo %FINITEDEPTH_DATA%

        .. code-tab:: powershell

            echo $env:FINITEDEPTH_DATA

.. _basic-example:

Basic example
-------------

Download :download:`config.yml <config.yml>` file in your local directory.
The contents of the file are:

.. literalinclude:: config.yml
    :language: yaml

The ``ref_path`` and ``coat_path`` are important parameters which specify
*reference image* and *target image(s)*. Note that the paths contain
environment variable which we already set.

The target image is analyzed using the reference image and other parameters
(``tempROI`` and ``substROI``) to yield visualized result. The result is
stored to the path specified by ``layer_visual``, which is an image file
in this example.

.. note::

    Refer to the :ref:`config-reference` page for an exhaustive description
    of every parameter in the configuration file.

Running the following command will generate ``result1.jpg`` which highlights
the coating layer region:

.. code-block:: bash

    finitedepth analyze config.yml

.. plot::
    :context: reset
    :caption: ``result1.jpg``

    import os, yaml, matplotlib.pyplot as plt
    from dipcoatimage.finitedepth import data_converter, Config
    with open(os.path.join("config.yml"), "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    (v,) = data.values()
    config = data_converter.structure(v, Config)
    coat = config.construct_coatinglayer(0)
    plt.axis("off")
    plt.imshow(coat.draw())
    plt.show()

Configuration file can also be ``JSON``.
Download :download:`config.json <config.json>` and run:

.. code-block:: bash

    finitedepth analyze config.json

.. literalinclude:: config.json
    :language: json

.. plot::
    :context: close-figs
    :caption: ``result2.jpg``

    import json
    with open(os.path.join("config.json"), "r") as f:
        data = json.load(f)
    (v,) = data.values()
    config = data_converter.structure(v, Config)
    coat = config.construct_coatinglayer(0)
    plt.axis("off")
    plt.imshow(coat.draw())
    plt.show()

.. note::

    To check supported formats for configuration file, run:

    .. code-block:: bash

        finitedepth analyze -h


Reference, substrate and coating layer
--------------------------------------

Configurations in the previous section are only minimum examples;
under the hood, there are more than meets the eye.

Change the ``config.yml`` as follows and run the analysis again:

.. literalinclude:: config-subst.yml
    :language: yaml

You will now have two additional files; ``ref1.jpg`` and ``subst1.jpg``.

.. plot::
    :context: close-figs
    :caption: ``ref1.jpg`` and ``subst1.jpg``

    with open(os.path.join("config.yml"), "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    (v,) = data.values()
    config = data_converter.structure(v, Config)
    coat = config.construct_coatinglayer(0)
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(coat.substrate.reference.draw())
    axes[0].axis("off")
    axes[1].imshow(coat.substrate.draw())
    axes[1].axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

One can discover that ``subst1.jpg`` is actually ``ref1.jpg`` cropped
by ``substrateROI``. Indeed, ``substrateROI`` specifies the
*substrate region* from the reference image, which is the red box in
``ref1.jpg``. Similary, ``templateROI`` specifies the *template region*
which helps locating the substrate region in the target image.

The :ref:`fundamental scheme <fundamentals>` is implemented as follows:

1. Reference image constructs *reference instance*
   (:class:`ReferenceBase`).
2. Reference instance constructs *substrate instance*
   (:class:`SubstrateBase`).
3. Substrate instance and target image construct *coating layer instance*
   (:class:`CoatingLayerBase`).
4. Coating layer instance defines the analysis result.

.. note::

    Actual analysis involves two additional types (:class:`ExperimentBase`
    and :class:`AnalysisBase`), but we spare the details in this section.


Specifying types
----------------

Previous configurations did not specify the instance types, defaulting
them to :class:`Reference`, :class:`Substrate`, and :class:`CoatingLayer`.
By using advanced types, more detailed analysis can be achieved.

Download :download:`config-rect.yml <config-rect.yml>` file in your local
directory.
The contents of the file are:

.. literalinclude:: config-rect.yml
    :language: yaml

Here, we specified the substrate type to :class:`RectSubstrate` and
coating layer type to :class:`RectLayerShape`. We also passed necessary
parameters, which are described in their docstrings..

The resulting coating layer instance is equipped with fancy visualization:

.. plot::
    :context: close-figs
    :caption: ``result3.jpg``

    with open(os.path.join("config-rect.yml"), "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    (v,) = data.values()
    config = data_converter.structure(v, Config)
    coat = config.construct_coatinglayer(0)
    plt.axis("off")
    plt.imshow(coat.draw())
    plt.show()

Also, analysis data of the substrate geometry and the coating layer shape are
saved in ``subst3.csv`` and ``result3.csv``:

.. csv-table:: subst3.csv
   :file: output/subst3.csv
   :header-rows: 1

.. csv-table:: result3.csv
   :file: output/result3.csv
   :header-rows: 1

The meaning of these data are described in the class docstrings of
:class:`RectSubstrate` and :class:`RectLayerShape`.

Likewise, by implementing your custom classes and specifying them in
configuration file, you can customize your analysis. Refer to :ref:`howto`
and :ref:`api` pages for more explanation.

Controlling visualization
-------------------------

Configuration file can define options to visualize the analysis result.

Change the ``config-rect.yml`` as follows and run the analysis again:

.. literalinclude:: config-rect2.yml
    :language: yaml

The :class:`RectLayerShape` now subtracts the substrate region from the
visualization result. Also, the uniform layer and the roughness samples
are no longer shown.

.. plot::
    :context: close-figs
    :caption: ``result3.jpg``

    with open(os.path.join("config-rect2.yml"), "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    (v,) = data.values()
    config = data_converter.structure(v, Config)
    coat = config.construct_coatinglayer(0)
    plt.axis("off")
    plt.imshow(coat.draw())
    plt.show()

Each reference, substrate, and coating layer class can define its own
visualization options.
