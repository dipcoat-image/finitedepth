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

You can notice that the file contains environment variable
``$FINITEDEPTH_DATA`` in paths. This allows us to load the sample data files
included in the package.

Running the following command will generate ``output/coat.jpg`` which
shows the coating layer region:

.. code-block:: bash

    finitedepth analyze config.yml

.. figure:: output/coat.jpg
   :align: center
   :width: 45%

   ``coat.jpg``

.. note::

    Refer to the :ref:`config-reference` page for an exhaustive description
    of every parameter in the configuration file.

Configuration file can also be ``JSON``.
The following :download:`config.json` file is equivalent to
:download:`config.yml`:

.. literalinclude:: config.json
    :language: json

.. note::

    To check all supported formats for configuration file, run:

    .. code-block:: bash

        finitedepth analyze -h


Core classes
------------

Configuration in the previous section are only a minimum example.
Under the hood, there is more than meets the eye.

Download :download:`config-extended.yml` as follows and run the analysis again:

.. literalinclude:: config-extended.yml
    :language: yaml

.. code-block:: bash

    finitedepth analyze config-extended.yml

This configuration has two additional fields compared to the previous file:
``analysis.parameters.ref_visual`` and ``analysis.parameters..subst_visual``.
As a result, it additionally generates ``output/ref.jpg``,

.. figure:: output/ref.jpg
   :align: center
   :figwidth: 45%

   ``ref.jpg``

and ``output/subst.jpg``.

.. figure:: output/subst.jpg
   :align: center
   :figwidth: 45%

   ``subst.jpg``

What you are seeing are visualization results of three core classes:
:ref:`reference class <howto-reference>`,
:ref:`substrate class <howto-substrate>`, and
:ref:`coating layer class <howto-coatinglayer>`

The :ref:`fundamental scheme <fundamentals>` of image analysis
is implemented as follows:

1. Reference instance stores the *reference image*.
2. Substrate instance stores the *substrate region*.
3. Coating layer instance stores the *target image* and detects
   the *coating layer region*.

Each class defines its own visualization method and analysis result, and
the configuration file can control how the results will be saved.

.. note::

    Actual analysis involves two more core classes:
    :ref:`experiment class <howto-experiment>` and
    :ref:`analysis class <howto-analysis>`.
    However, we spare the details in this section.


Specifying types
----------------

Previous configurations did not specify the core class types, defaulting
them to :class:`Reference`, :class:`Substrate`, and :class:`CoatingLayer`.
By using advanced types, more detailed analysis can be achieved.

Download :download:`config-rect.yml` file. The contents of the file are:

.. literalinclude:: config-rect.yml
    :language: yaml

Here, the core classes are set to the following types:

* Reference type is not specified, thus defaults to :class:`Reference`.
* Substrate type is set to :class:`RectSubstrate`.
* Coating layer type is set to :class:`RectLayerShape`.

The new types require additional parameters, which are also specified
in the configuration file.

The resulting coating layer instance is equipped with fancy visualization:

.. figure:: output/rectcoat.jpg
   :align: center
   :figwidth: 45%

   ``rectcoat.jpg``

Also, analysis result of the substrate and the coating layer are
saved in ``rectsubst.csv`` and ``rectcoat.csv``:

.. csv-table:: rectsubst.csv
   :file: output/rectsubst.csv
   :header-rows: 1

.. csv-table:: rectcoat.csv
   :file: output/rectcoat.csv
   :header-rows: 1

Likewise, you can implement your own classes equipped with
custom visualization and analysis algorithms, and specify them in
configuration file to perform customized analysis.
Refer to :ref:`howto` and :ref:`api` pages for more details.

Controlling visualization
-------------------------

Configuration file can control visualization result.

Download :download:`config-rect2.yml` file. The contents of the file are:

.. literalinclude:: config-rect2.yml
    :language: yaml

This configuration has two additional fields compared to the previous file:
``coatinglayer.draw_options`` and ``coatinglayer.deco_options``.
As a result, the visualization result is now different.

.. figure:: output/rectcoat2.jpg
   :align: center
   :figwidth: 45%

   ``rectcoat2.jpg``

The additional fields are purely cosmetic and does not affect the
analysis result, which is determined by ``parameters``.

.. csv-table:: rectcoat2.csv
   :file: output/rectcoat2.csv
   :header-rows: 1

Each reference, substrate, and coating layer class can define its own
visualization options.
