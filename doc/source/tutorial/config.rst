.. _config-example:

Analyzing from configuration file
=================================

Systematic analysis can be achieved by specifying the analysis parameters
in configuration file.

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

Basic example
-------------

Download :download:`config1.yml <./config1.yml>` file in your local directory.
The contents of the file is:

.. literalinclude:: ./config1.yml
    :language: yaml

The important parameters in the configuration file are ``ref_path`` and
``coat_path``, which are the paths to reference image and target image(s).
Note that the paths contain environment variable which we already set.

For reference image, *template ROI* (``tempROI``) and *substrate ROI*
(``substROI``) are specified. These parameters are explained in
:class:`ReferenceBase`.

Finally, the visualization result is specified in ``layer_visual``.
As a result, the coating layer image will be saved as an image file.

.. note::

    Refer to the :ref:`config-reference` page for an exhaustive description
    of every parameter in the configuration file.

:download:`config2.json <./config2.json>`

.. literalinclude:: ./config2.json
    :language: json


:download:`config3.yml <./config3.yml>`

.. literalinclude:: ./config3.yml
    :language: yaml
