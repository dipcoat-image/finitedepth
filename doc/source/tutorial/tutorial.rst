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

We begin with a brief example which performs analysis using command-line and
configuration file.

Download :download:`config.yml` file in your local directory.
The contents of this configuration file are:

.. literalinclude:: config.yml
    :language: yaml

You can notice that the file contains environment variable ``$FINITEDEPTH_DATA``
in paths. This allows us to load the sample data files included in the package.

Running the following command will generate ``output/coat.jpg`` which
highlights the coating layer region:

.. code-block:: bash

    finitedepth analyze config.yml

.. figure:: output/coat.jpg
   :align: center

   ``coat.jpg``

.. note::

    Refer to the :ref:`config-reference` page for an exhaustive description
    of every parameter in the configuration file.

Configuration file can also be ``JSON``.
The following :download:`config.json` file is equivalent to :download:`config.yml`:

.. literalinclude:: config.json
    :language: json

.. note::

    To check all supported formats for configuration file, run:

    .. code-block:: bash

        finitedepth analyze -h


Reference, substrate and coating layer
--------------------------------------

Configurations in the previous section are only minimum examples.
Under the hood, there are more than meets the eye.

Download :download:`config-extended.yml` as follows and run the analysis again:

.. literalinclude:: config-extended.yml
    :language: yaml

.. code-block:: bash

    finitedepth analyze config-extended.yml

This file has 2 additional fields compared to :download:`config.yml`:
``ref_visual`` and ``subst_visual``. As a result, it additionally generates
``output/ref.jpg``

.. figure:: output/ref.jpg
   :align: center
   :figwidth: 45%

   ``ref.jpg``

and ``output/subst.jpg``.

.. figure:: output/subst.jpg
   :align: center
   :figwidth: 45%

   ``subst.jpg``

One can discover that ``subst.jpg`` is actually ``ref.jpg`` cropped
by ``substrateROI``. Indeed, ``substrateROI`` specifies the
*substrate region* from the reference image, which is the red box in
``ref.jpg``. Similary, ``templateROI`` specifies the *template region*
which helps locating the substrate region in the target image.

The :ref:`fundamental scheme <fundamentals>` is implemented as follows:

1. Reference image constructs *reference instance*
   (inherits :class:`ReferenceBase`).
2. Reference instance constructs *substrate instance*
   (inherits :class:`SubstrateBase`).
3. Substrate instance and target image construct *coating layer instance*
   (inherits :class:`CoatingLayerBase`), which defines analysis result.

.. note::

    Actual analysis involves *experiment class* (inherits
    :class:`ExperimentBase`) and *analysis class* (inherits
    :class:`AnalysisBase`), but we spare the details in this section.
    Refer to :ref:`howto-runtime` page and class docstrings.


Specifying types
----------------

Previous configurations did not specify the instance types, defaulting
them to :class:`Reference`, :class:`Substrate`, and :class:`CoatingLayer`.
By using advanced types, more detailed analysis can be achieved.

Download :download:`config-rect.yml` file. The contents of the file are:

.. literalinclude:: config-rect.yml
    :language: yaml

Here, we specified the substrate type to :class:`RectSubstrate` and
coating layer type to :class:`RectLayerShape`. We also passed necessary
parameters, which are described in their docstrings..

The resulting coating layer instance is equipped with fancy visualization:

.. figure:: output/rectcoat.jpg
   :align: center

   ``rectcoat.jpg``

Also, analysis data of the substrate geometry and the coating layer shape are
saved in ``rectsubst.csv`` and ``rectcoat.csv``:

.. csv-table:: subst.csv
   :file: output/subst.csv
   :header-rows: 1

.. csv-table:: rectcoat.csv
   :file: output/rectcoat.csv
   :header-rows: 1

The meaning of these data are described in the class docstrings of
:class:`RectSubstrate` and :class:`RectLayerShape`.

Likewise, by implementing your custom classes and specifying them in
configuration file, you can customize your analysis. Refer to :ref:`howto`
and :ref:`api` pages for more explanation.

Controlling visualization
-------------------------

Configuration file can define options to visualize the analysis result.

Download :download:`config-rect2.yml` file. The contents of the file are:

.. literalinclude:: config-rect2.yml
    :language: yaml

The visualization result is now different.

.. figure:: output/rectcoat2.jpg
   :align: center

   ``rectcoat2.jpg``

The new parameters are purely cosmetic and does not modify the analysis data.

.. csv-table:: rectcoat2.csv
   :file: output/rectcoat2.csv
   :header-rows: 1

Each reference, substrate, and coating layer class can define its own
visualization options.
